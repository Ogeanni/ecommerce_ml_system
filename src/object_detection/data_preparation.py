""" Data Preparation for Object Detection """

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

class ObjectDetectionDataPreparator:
    """ Prepare dataset for YOLO training """
    def __init__(self, dataset_name="coco_subset"):
        """
        Initialize data preparator
        
        Args:
            dataset_name: Name of dataset
        """
        self.dataset_name = dataset_name
        self.data_dir = Path("data/object_detection")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = []

    def create_yolo_structure(self):
        """ Create YOLO directory structure """
        print("Creating YOLO directory structure...")

        for split in ["train", "val", "test"]:
            (self.data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

            print(" Directory structure created")

    def download_coco_subset(self, categories: List[str] = None, max_images_per_category=500):
        """
        Download COCO dataset subset
    
        Args:
        categories: List of category names to include
        max_images_per_category: Maximum images per category
        """
        print("\n" + "="*80)
        print("DOWNLOADING COCO SUBSET")
        print("="*80)

        # Default e-commerce relevant categories
        if categories is None:
            categories = [
                'bottle', 'cup', 'bowl', 'laptop', 'mouse', 'keyboard',
                'cell phone', 'book', 'clock', 'vase', 'scissors',
                'backpack', 'handbag', 'suitcase', 'shoe', 'chair',
                'couch', 'bed', 'dining table', 'tv'
                ]
            print(f"\nCategories to download: {len(categories)}")
            print(f"Max images per category: {max_images_per_category}")

        try:
            import fiftyone as fo
            import fiftyone.zoo as foz

            print("\nDownloading COCO dataset subset...")
            print("This may take a while on first run...")

            # Create a mapping for class names to ensure consistency
            self.class_names = categories

            # Map our splits to FiftyOne splits
            # YOLOv5 expects these exact folder names: train, val, test
            splits_config = [
                {"yolo_split": "train", 
                 "fo_split": "train",
                 "max_samples": max_images_per_category * len(categories)
                 },
                {"yolo_split": "val_temp", # Temporary - will split into val and test
                 "fo_split": "validation", 
                 "max_samples": max_images_per_category * len(categories) // 2
                 }
                ]
            
            # Temporary validation data for splitting
            val_temp_images = []
            val_temp_labels = []

            for config in splits_config:
                yolo_split = config["yolo_split"]
                fo_split = config["fo_split"]
                max_samples = config["max_samples"]

                print(f"\n{'='*60}")
                print(f"Processing {yolo_split} split (using COCO {fo_split})...")
                print(f"Max samples: {max_samples}")
                print(f"{'='*60}")

                # Using unique dataset name to avoid conflicts
                dataset_name = f"coco_subset_{yolo_split}_{self.dataset_name}"

                # Delete existing dataset if it exists
                if dataset_name in fo.list_datasets():
                    print(f"Deleting existing dataset: {dataset_name}")
                    fo.delete_dataset(dataset_name)

                # Load dataset from zoo
                dataset = foz.load_zoo_dataset(
                    "coco-2017",
                    split=fo_split,
                    label_types=["detections"],
                    classes=categories,
                    max_samples=max_samples,
                    dataset_name=dataset_name
                    )
                
            
                print(f"Loaded {len(dataset)} images from FiftyOne cache")
            
                if len(dataset) == 0:
                    raise ValueError(
                    f"No images loaded for split {yolo_split}. Check your category names!\n"
                    f"Categories: {categories}"
                    )
                
                # Inspect dataset to find the correct label field
                print("\nDataset schema:")
                print(dataset)
                sample = dataset.first()
                print(f"\nSample fields: {sample.field_names}")

                # Find the detection field (usually 'ground_truth' or 'detections')
                detection_field = None
                for field_name in sample.field_names:
                    field_value = sample[field_name]
                    if isinstance(field_value, fo.Detections):
                        detection_field = field_name
                        print(f"\n Found detections field: '{detection_field}'")
                        break
            
                if detection_field is None:
                    raise ValueError("Could not find detections field in dataset!")
                
                # Determine output split name
                output_split = "val_temp" if yolo_split == "val_temp" else yolo_split

                # Create output directories
                images_dir = self.data_dir / "images" / yolo_split
                labels_dir = self.data_dir / "labels" / yolo_split
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)

                print(f"Exporting to: {self.data_dir}")
                print(f"  Images: {images_dir}")
                print(f"  Labels: {labels_dir}")

                # Export to YOLOv5 format
                dataset.export(
                    export_dir=str(self.data_dir),
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field=detection_field,  # Specify the label field
                    split=output_split,
                    classes=categories,  # Ensure consistent class ordering
                    )

                # Verify images were exported
                exported_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                exported_labels = list(labels_dir.glob("*.txt"))
            
                print(f"\n Exported {len(exported_images)} images")
                print(f" Exported {len(exported_labels)} label files")
            
                if len(exported_images) == 0:
                    print(f"  WARNING: No images exported for {yolo_split}!")
                
                # Store temporary val data for splitting
                if yolo_split == "val_temp":
                    val_temp_images = exported_images
                    val_temp_labels = exported_labels

                # # Clean up the FiftyOne dataset if exist to save memory
                if dataset_name in fo.list_datasets():
                    try:
                        fo.delete_dataset(dataset_name)
                        print(f" Cleaned up FiftyOne dataset: {dataset_name}")
                    except Exception as e:
                        print(f"  Could not delete dataset {dataset_name}: {e}")
            
            if val_temp_images:
                print(f"\n{'='*60}")
                print("Splitting validation data into val and test sets...")
                print(f"{'='*60}")

                import random
                random.seed(42)

                # Shuffle images
                indices = list(range(len(val_temp_images)))
                random.shuffle(indices)

                # Calculate split point (70% val, 30% test)
                split_idx = int(len(indices) * 0.7)
                
                val_indices = indices[:split_idx]
                test_indices = indices[split_idx:]

                # Create val and test directories
                for split_name in ["val", "test"]:
                    (self.data_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
                    (self.data_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

                # Move val images
                for idx in val_indices:
                    img_path = val_temp_images[idx]
                    label_path = self.data_dir / "labels" / "val_temp" / (img_path.stem + '.txt')

                    new_img = self.data_dir / "images" / "val" / img_path.name
                    new_label = self.data_dir / "labels" / "val" / (img_path.stem + '.txt')

                    shutil.move(str(img_path), str(new_img))
                    if label_path.exists():
                        shutil.move(str(label_path), str(new_label))

                # Move test images
                for idx in test_indices:
                    img_path = val_temp_images[idx]
                    label_path = self.data_dir / "labels" / "val_temp" / (img_path.stem + '.txt')

                    new_img = self.data_dir / "images" / "test" / img_path.name
                    new_label = self.data_dir / "labels" / "test" / (img_path.stem + '.txt')

                    shutil.move(str(img_path), str(new_img))
                    if label_path.exists():
                        shutil.move(str(label_path), str(new_label))
                
                # Remove temporary directories
                shutil.rmtree(self.data_dir / "images" / "val_temp", ignore_errors=True)
                shutil.rmtree(self.data_dir / "labels" / "val_temp", ignore_errors=True)

                print(f" Created validation set: {len(val_indices)} images")
                print(f" Created test set: {len(test_indices)} images")


            print("\n" + "="*80)
            print(" COCO SUBSET DOWNLOAD COMPLETE")
            print("="*80)
        
            return True

        except ImportError:
            print("\n fiftyone not installed.")
            print("Install with: pip install fiftyone")
            print("\nAlternative: Download COCO manually and convert to YOLO format")
            return False
        except Exception as e:
            print(f"\n Error downloading dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def convert_coco_to_yolo(self,
                             coco_json_path: str,
                             image_dir: str,
                             output_dir: str,
                             split: str = "train"):
        """
        Convert COCO format annotations to YOLO format
        
        Args:
            coco_json_path: Path to COCO JSON file
            image_dir: Directory containing images
            output_dir: Output directory for YOLO format
            split: Dataset split (train/val/test)
        """

        print(f"\nConverting COCO to YOLO format for {split} split...")

        # Load COCO annotations
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        # Create category mapping
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        category_ids = sorted(categories.keys())
        category_names = [categories[cid] for cid in category_ids]

        # Create output directories
        images_out = Path(output_dir) / "images" / split
        labels_out = Path(output_dir) / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Group annotations by image
        image_annotation = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image_annotation:
                image_annotation[image_id] = []
            image_annotation[image_id].append(ann)

        # Convert each image
        converted_count = 0
        for image_info in tqdm(coco_data["images"], desc=f"Converting {split}"):
            image_id = image_info["id"]
            image_filename = image_info["file_name"]
            image_width = image_info["width"]
            image_height = image_info["height"]

            # Copy image
            src_image = Path(image_dir) / image_filename
            dst_image = images_out / image_filename

            if not src_image.exists():
                continue

            shutil.copy(src_image, dst_image)

            # Convert annotations to YOLO format
            if image_id in image_annotation:
                yolo_annotations = []

                for ann in image_annotation[image_id]:
                    # Get category index (0-based)
                    category_idx = category_ids.index(ann["category_id"])

                    # COCO bbox: [x, y, width, height]
                    bbox = ann["bbox"]
                    x_center = (bbox[0] + bbox[2] / 2) / image_width
                    y_center = (bbox[1] + bbox[3] / 2) / image_height
                    width = bbox[2] / image_width
                    height = bbox[3] / image_height

                    # YOLO format: class x_center y_center width height (normalized)
                    yolo_annotations.append(
                        f"{category_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                # Write YOLO label file
                label_filename = Path(image_filename).stem + ".txt"
                label_path = labels_out / label_filename

                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_annotations))

                converted_count += 1

        print(f" Converted {converted_count} images for {split} split")
        return category_names
    
    def create_dataset_yaml(self, class_names: List[str], yaml_path: str = None):
        """
        Create YAML configuration file for YOLOv5
        
        Args:
            class_names: List of class names
            yaml_path: Path to save YAML file
        """
        if yaml_path is None:
            yaml_path = self.data_dir / "dataset.yaml"

        # Use absolute paths
        train_path = (self.data_dir / "images" / "train").absolute()
        val_path = (self.data_dir / "images" / "val").absolute()
        test_path = (self.data_dir / "images" / "test").absolute()

        # Create YAML content
        yaml_content = {
            "path": str(self.data_dir.absolute()),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "nc": len(class_names),
            "names": class_names
        }

        # Write YAML file
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\n Created dataset YAML: {yaml_path}")
        print(f"   Number of classes: {len(class_names)}")
        print(f"   Classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

        # Verify paths
        for split, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
            if path.exists():
                images = list(path.glob('*.[jp][pn]g'))
                print(f"    {split}: {len(images)} images")
            else:
                print(f"     {split}: Path does not exist: {path}")

        return yaml_path
    

    def visualize_sample(self,
                         image_path: str,
                         label_path: str,
                         class_names: List[str],
                         save_path: str = None):
        """
        Visualize image with bounding boxes
        
        Args:
            image_path: Path to image
            label_path: Path to YOLO label file
            class_names: List of class names
            save_path: Path to save visualization
        """
        # Load image
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Load labels
        bboxes = []
        if Path(label_path).exists():
            with open(label_path, "r") as f:
                for line in f:
                    values = line.strip().split()
                    class_id = int(values[0])
                    x_center = float(values[1]) * img_width
                    y_center = float(values[2]) * img_height
                    width = float(values[3]) * img_width
                    height = float(values[4]) * img_height

                    # Convert to corner coordinates
                    x1 = x_center - width/2
                    y1 = y_center - height/2

                    bboxes.append({
                        "class": class_names[class_id],
                        "bbox": [x1, y1, width, height]
                    })

        # Plot
        fig, ax = plt.subplots(1, figsize=(12,8))
        ax.imshow(image)

        # Draw bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))

        for bbox_info in bboxes:
            class_name = bbox_info["class"]
            class_idx = class_names.index(class_name)
            bbox = bbox_info["bbox"]

            # Create rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2,
                edgecolor=colors[class_idx],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                bbox[0], bbox[1] - 5,
                class_name,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=colors[class_idx], alpha=0.7, edgecolor='none')
            )
        
        ax.axis('off')
        plt.title(f'Image with {len(bboxes)} objects', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Visualization saved to {save_path}")
        
        plt.show()
        plt.close()

    def analyze_dataset(self, class_names: List[str]):
        """ 
        Analyze dataset statistics
        
        Args:
            class_names: List of class names
        """
        print("\n" + "="*80)
        print("DATASET ANALYSIS")
        print("="*80)

        for split in ["train", "val", "test"]:
            image_dir = self.data_dir / "images" / split
            label_dir = self.data_dir / "labels" / split

            if not image_dir.exists():
                continue

            # Count images
            image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            num_images = len(image_files)


            # Count objects per class
            class_counts = {name: 0 for name in class_names}
            total_objects = 0

            for label_file in label_dir.glob("*.txt"):
                with open(label_file, "r") as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
                            total_objects += 1
                        else:
                            print(f" Skipping invalid class_id {class_id} in {label_file.name}")
                        
            
            print(f"\n{split.upper()} SET:")
            print(f"  Images: {num_images}")
            print(f"  Total objects: {total_objects}")
            print(f"  Avg objects per image: {total_objects / num_images if num_images > 0 else 0:.2f}")
            
            # Print class distribution
            print(f"\n  Class distribution:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"    {class_name:<20}: {count:>5} ({count/total_objects*100:>5.1f}%)")

def prepare_coco_subset():
    """ Convenience function to prepare COCO subset """
    preparator = ObjectDetectionDataPreparator("coco-subset")

    # Create directory structure
    preparator.create_yolo_structure()

    # Download COCO subset (using fiftyone)
    success = preparator.download_coco_subset(
        categories=['bottle', 'cup', 'laptop', 'mouse', 'keyboard','cell phone', 'book', 'backpack', 'handbag'],
        max_images_per_category=500)
    
    if not success:
        print("\nPlease install fiftyone or manually download COCO dataset")
        return None
    
    # Get class names from data
    # This would be extracted from the downloaded data
    class_names =['bottle', 'cup', 'laptop', 'mouse', 'keyboard',
                   'cell phone', 'book', 'backpack', 'handbag']
    
    # Create dataset YAML
    yaml_path = preparator.create_dataset_yaml(class_names)

    # Analyze dataset
    preparator.analyze_dataset(class_names)

    # Visualize sample
    images_dir = preparator.data_dir / "images" / "val"
    labels_dir = preparator.data_dir / "labels" / "val"

    image_files = list(images_dir.glob('*.jpg'))
    if image_files:
        sample_image = image_files[0]
        sample_label = labels_dir / (sample_image.stem + '.txt')
        
        preparator.visualize_sample(
            str(sample_image),
            str(sample_label),
            class_names,
            save_path='results/object_detection/sample_annotation.png'
        )

    return yaml_path

if __name__ == "__main__":
    prepare_coco_subset()
