""" Training pipeline for object detection using YOLOv5 """

import torch
import yaml
from pathlib import Path
import subprocess
import sys

def resolve_root_dir(project_name: str) -> Path:
    """
    Resolve root directory for local vs Colab + Google Drive
    """
    gdrive_root = Path("/content/drive/MyDrive")

    if gdrive_root.exists():
        # Running on Colab with Google Drive mounted
        return gdrive_root / project_name
    else:
        # Running locally
        return Path(__file__).resolve().parent # ecommerce_ml_system root


class ObjectDetectionTrainer:
    """ Train YOLOv5 object detection models """
    def __init__(self,
                 data_yaml: str,
                 model_size: str = "s",
                 project_name: str = "object_detection"):
        """
        Initialize trainer
        
        Args:
            data_yaml: Path to dataset YAML file
            model_size: YOLOv5 model size (n, s, m, l, x)
            project_name: Project name for saving results
        """
        self.root_dir = resolve_root_dir("ecommerce_ml_system")
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.project_name = project_name

        # YOLOv5 directory
        self.yolov5_dir = (self.root_dir / "models" / "object_detection" / "yolov5")

        if not self.yolov5_dir.exists():
            print(f"  YOLOv5 not found at {self.yolov5_dir}")
            print(" Attempting to clone YOLOv5...")
            self._setup_yolov5()

        if not self.yolov5_dir.exists():
            raise FileNotFoundError(
                f"YOLOv5 not found at {self.yolov5_dir}. "
                "Please clone it first: git clone https://github.com/ultralytics/yolov5.git"
            )
        
        # Output directory
        self.output_dir = (self.root_dir / "models" / "object_detection" / "runs" / project_name)

        # Validate dataset YAML
        self.validate_dataset()

        print(f"   Trainer initialized")
        print(f"   Data YAML: {self.data_yaml}")
        print(f"   Model size: yolov5{self.model_size}")
        print(f"   Output directory: {self.output_dir}")

    
    def _setup_yolov5(self):
        """Clone and setup YOLOv5 repository"""
        try:
            # Create parent directory
            self.yolov5_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Clone YOLOv5
            print("Cloning YOLOv5 repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ultralytics/yolov5.git", str(self.yolov5_dir)],
                check=True,
                cwd=self.yolov5_dir.parent
            )
            print(" YOLOv5 cloned successfully")
            
        except subprocess.CalledProcessError as e:
            print(f" Error setting up YOLOv5: {e}")
            print("\nPlease manually clone YOLOv5:")
            print(f"  cd {self.yolov5_dir.parent}")
            print("  git clone https://github.com/ultralytics/yolov5.git")
        except Exception as e:
            print(f" Unexpected error: {e}")

    def validate_dataset(self):
        """Validate dataset YAML and check if all paths exist"""
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml}")
        print("\n" + "="*80)
        print("VALIDATING DATASET")
        print("="*80)

        # Load YAML
        with open(self.data_yaml, "r") as f:
            data = yaml.safe_load(f)

        print(f"\nDataset configuration:")
        print(f"  Number of classes: {data.get('nc', 'NOT SET')}")
        print(f"  Class names: {data.get('names', 'NOT SET')}")

        # Check required fields
        required_fields = ["train", "val", "test", "nc", "names"]
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            raise ValueError(f"Missing required fields in YAML: {missing_fields}")
        
        # Validate paths
        print(f"\nValidating paths...")

        # Handle both absolute and relative paths
        yaml_dir = self.data_yaml.parent

        for split in ["train", "val", "test"]:
            path_str = data[split]
            path = Path(path_str)

            # If relative path, make it relative to YAML file location
            if not path.is_absolute():
                path = yaml_dir/path

            if path.exists():
                # Count files
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                images = []
                for ext in image_extensions:
                    images.extend(list(path.glob(f"*{ext}")))
                
                print(f"   {split}: {path} ({len(images)} images)")

                if len(images) == 0:
                    raise ValueError(f"No images found in {split} directory: {path}")
            else:
                print(f"   {split}: {path} - NOT FOUND!")

                # Try to suggest fix
                print(f"\n  Looking for directory...")
                print(f"  YAML location: {self.data_yaml}")
                print(f"  Looking for: {path}")

                # Check if images/train or images/val exists
                alt_path = yaml_dir / 'images' / split
                if alt_path.exists():
                    print(f"    Found: {alt_path}")
                    print(f"  Update your dataset.yaml to use:")
                    print(f"    {split}: images/{split}")
                raise FileNotFoundError(f"{split} directory not found: {path}")
        
        print("\n Dataset validation complete!")


    def train(self,
              epochs: int=100,
              batch_size: int=16,
              img_size: int=416,
              device: str="0",
              workers: int=8,
              resume: bool=False,
              **kwargs):
        """
        Train YOLOv5 model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            device: GPU device (0, 1, 2, etc.) or 'cpu'
            workers: Number of dataloader workers
            resume: Resume from last checkpoint
            **kwargs: Additional arguments for train.py
        """

        print("\n" + "="*80)
        print("TRAINING YOLOV5 OBJECT DETECTION MODEL")
        print("="*80)

        # Check if CUDA is available
        if device != "cpu" and not torch.cuda.is_available():
            print("  CUDA not available, using CPU")
            device = "cpu"

        print(f"\nTraining parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Device: {device}")
        print(f"  Model: yolov5{self.model_size}")

        # Build command
        train_script = "train.py"

        cmd = [
            sys.executable, str(train_script),
            '--data', str(self.data_yaml.resolve()),
            '--weights', f'yolov5{self.model_size}.pt',  # Pretrained weights
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--imgsz', str(img_size),
            '--device', str(device),
            '--workers', str(workers),
            '--project', str(self.output_dir),
            '--name', 'train',
            '--exist-ok',
        ]

        # Add additional arguments
        for key, value in kwargs.items():
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        if resume:
            cmd.append('--resume')

        print(f"\nRunning command:")
        print(' '.join(cmd))
        print()

        # Run training
        try:
            subprocess.run(cmd, check=True, cwd=self.yolov5_dir)
            print("\n Training complete!")
            print(f"   Results saved to: {self.output_dir / 'train'}")
        
        except subprocess.CalledProcessError as e:
            print(f"\n Training failed with error: {e}")
            return False
        
        return True
    
    def get_best_weights_path(self):
        """ Get path to best model weights """
        weights_path = self.output_dir / "train" / "weights" / "best.pt"
        if weights_path.exists():
            return weights_path
        return None
    

def train_object_detector(data_yaml: str,
                          model_size: str = "s",
                          epochs: int = 100,
                          batch_size: int = 16):
    """ 
    Convenience function to train object detector
    
    Args:
        data_yaml: Path to dataset YAML
        model_size: Model size (n, s, m, l, x)
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Path to best weights
    """
    trainer = ObjectDetectionTrainer(data_yaml, model_size)

    success = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        img_size=640,
        device="0" if torch.cuda.is_available() else "cpu"
    )

    if success:
        return trainer.get_best_weights_path()
    
    return None

if __name__ == "__main__":
    # Example training
    data_yaml = "data/object_detection/dataset.yaml"

    if Path(data_yaml).exists():
        train_object_detector(
            data_yaml=data_yaml,
            model_size="s",
            epochs=100,
            batch_size=16
        )

    else:
        print(f"Dataset YAML not found: {data_yaml}")
        print("Run data_preparation.py first")