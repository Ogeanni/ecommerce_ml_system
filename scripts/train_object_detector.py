""" Complete end-to-end training script for object detection """

import sys

from pathlib import Path
import yaml
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import get_task_dirs

dirs = get_task_dirs("object_detection")

MODEL_DIR = dirs["saved_models"]
CHECKPOINT_DIR = dirs["checkpoints"]
RESULTS_DIR = dirs["results"]
LOGS_DIR = dirs["logs"]

from src.object_detection.data_preparation import ObjectDetectionDataPreparator, prepare_coco_subset
from src.object_detection.train import ObjectDetectionTrainer
from src.object_detection.evaluate import ObjectDetectionEvaluator
from src.object_detection.detect import ObjectDetector

print("="*80)
print("OBJECT DETECTION - COMPLETE TRAINING PIPELINE")
print("="*80)

# ==================== CONFIGURATION ====================
CONFIG = {
    # Data
    'dataset_name': 'coco_subset',
    'categories': [
        'bottle', 'cup', 'laptop', 'mouse', 'keyboard',
        'cell phone', 'book', 'backpack', 'handbag', 'suitcase'
    ],
    'max_images_per_category': 500,
    
    # Model
    'model_size': 's',  # n (nano), s (small), m (medium), l (large), x (xlarge)
    
    # Training
    'epochs': 100,
    'batch_size': 16,
    'img_size': 416,
    'patience': 50,  # Early stopping patience
    
    # Evaluation
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    if isinstance(value, list) and len(value) > 5:
        print(f"  {key}: {value[:5]}... ({len(value)} total)")
    else:
        print(f"  {key}: {value}")

# ==================== STEP 1: PREPARE DATA ====================
print("\n" + "="*80)
print("STEP 1: DATA PREPARATION")
print("="*80)

data_dir = Path("data/object_detection")
dataset_yaml = data_dir / "dataset.yaml"

def check_dataset_valid():
    """Check if dataset has sufficient images in all splits"""
    if not dataset_yaml.exists():
        print(" dataset.yaml not found")
        return False
    
    # Load YAML to check class count
    try:
        with open(dataset_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Check if classes match config
        if yaml_data.get('names') != CONFIG['categories']:
            print("  Class names in YAML don't match config")
            return False
    except:
        print(" Could not read dataset.yaml")
        return False
    
    # Check minimum image counts for each split
    min_images = {
        'train': 1500,  # Expect at least 1500 train images with 500/category
        'val': 300,     # Expect at least 300 val images
        'test': 100     # Expect at least 100 test images
    }
    
    print("\nChecking existing dataset:")
    all_valid = True
    
    for split, min_count in min_images.items():
        split_dir = data_dir / "images" / split
        if not split_dir.exists():
            print(f"   {split}: directory not found")
            all_valid = False
            continue
        
        num_images = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))
        
        if num_images < min_count:
            print(f"    {split}: {num_images} images (expected >= {min_count})")
            all_valid = False
        else:
            print(f"  {split}: {num_images} images")
    
    return all_valid

# Check if we need to prepare data
need_preparation = not check_dataset_valid() or CONFIG.get('force_redownload', False)

if need_preparation:
    print("\n Preparing dataset...")
    
    # Clean existing data if forcing redownload
    if CONFIG.get('force_redownload', False):
        print(" Force redownload enabled - cleaning existing data...")
        import shutil
        for dir_path in [(data_dir / "images"), (data_dir / "labels")]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  Deleted: {dir_path}")
        if dataset_yaml.exists():
            dataset_yaml.unlink()
            print(f"  Deleted: {dataset_yaml}")
    
    # Continue with data preparation...
    preparator = ObjectDetectionDataPreparator(CONFIG["dataset_name"])

    # Create directory structure
    preparator.create_yolo_structure()

    # Download COCO subset
    print("\nDownloading COCO subset...")
    print("  Note: This requires 'fiftyone' package")
    print("   Install with: pip install fiftyone")

    try:
        data_success = preparator.download_coco_subset(
            categories=CONFIG["categories"],
            max_images_per_category=CONFIG["max_images_per_category"])
        
        if data_success:
            # Check image counts
            for split in ["train", "val", "test"]:
                split_dir = data_dir / "images" / split
                num_images = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))
                print(f"Downloaded {num_images} images in {split}")
            # Create dataset YAML
            dataset_yaml = preparator.create_dataset_yaml(CONFIG["categories"])

            # Analyze dataset
            preparator.analyze_dataset(CONFIG["categories"])

            # Visualize sample
            images_dir = data_dir / "images" / "val"
            labels_dir = data_dir / "labels" / "val"

            image_files = list(images_dir.glob("*.jpg"))
            if image_files:
                sample_image = image_files[0]
                sample_label = labels_dir / f"{sample_image.stem}.txt"

                preparator.visualize_sample(
                    str(sample_image),
                    str(sample_label),
                    CONFIG["categories"],
                    save_path = "results/object_detection/sample_annotation.png"
                )
            
        else:
            print("\n Data preparation failed")
            print("Please download COCO manually or use a different dataset")
            sys.exit(1)

    except ImportError:
        print("\n fiftyone package not found")
        print("Install with: pip install fiftyone")
        print("\nAlternatively, manually prepare your dataset in YOLO format")
        sys.exit(1)

else:
    print(f"\n Dataset already prepared: {dataset_yaml}")

    # Load class names
    with open(dataset_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    print(f"   Classes: {data_config['names']}")

# ==================== STEP 2: TRAIN MODEL ====================
print("\n" + "="*80)
print("STEP 2: TRAINING MODEL")
print("="*80)

trainer = ObjectDetectionTrainer(
    data_yaml=str(dataset_yaml),
    model_size=CONFIG["model_size"],
    project_name="ecommerce_quality_control"
)

print(f"\nTraining YOLOv5{CONFIG['model_size']} for {CONFIG['epochs']} epochs...")

# Train model

training_success = trainer.train(
    epochs=CONFIG["epochs"],
    batch_size=CONFIG["batch_size"],
    img_size=CONFIG["img_size"],
    device="cpu",
    workers=2,
    patience=CONFIG["patience"],
    save_period = 10,
    cache = False,

)

if not training_success:
    print("\n Training failed")
    sys.exit(1)

# Get best weights path

best_weights = trainer.get_best_weights_path()
if best_weights:
    print(f"\n Training complete!")
    print(f"   Best weights: {best_weights}")
else:
    print("\n  Training completed but best weights not found")
    print("   Check the training output directory")

# ==================== STEP 3: EVALUATE MODEL ====================
print("\n" + "="*80)
print("STEP 3: EVALUATING MODEL")
print("="*80)

if best_weights and best_weights.exists():
    evaluator = ObjectDetectionEvaluator(
        weights_path=str(best_weights),
        data_yaml=str(dataset_yaml),
        conf_threshold=CONFIG["conf_threshold"],
        iou_threshold=CONFIG["iou_threshold"]
    )

    # Generate comprehensive evaluation report
    metrics = evaluator.generate_full_report(
    output_dir="results/object_detection/evaluation"
)
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    print(f"\nTest Set Performance:")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  mAP@0.5: {metrics.get('map50', 0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.get('map', 0):.4f}")

else:
    print("\n  Skipping evaluation - no weights found")


# ==================== STEP 4: TEST INFERENCE ====================
print("\n" + "="*80)
print("STEP 4: TESTING INFERENCE")
print("="*80)

if best_weights and best_weights.exists():
    detector = ObjectDetector(
        weights_path=str(best_weights),
        data_yaml=str(dataset_yaml),
        conf_threshold=CONFIG["conf_threshold"],
        iou_threshold=CONFIG["iou_threshold"]
    )

    # Test on sample images
    test_images_dir = data_dir / 'images' / 'test'

    if test_images_dir.exists():
        test_images = list(test_images_dir.glob('*.jpg'))[:5]
    
        if test_images:
            print(f"\nTesting on {len(test_images)} sample images...")
        
            for i, img_path in enumerate(test_images, 1):
                print(f"\n{'='*60}")
                print(f"Sample {i}/{len(test_images)}: {img_path.name}")
                print(f"{'='*60}")
            
                detections = detector.detect_and_visualize(
                    str(img_path),
                    save_path=f'results/object_detection/inference_sample_{i}.png')
            
                print(f"\nDetection summary:")
                print(f"  Total objects: {detections['num_detections']}")
            
                # Count by class
                class_counts = {}
                for det in detections['detections']:
                    class_name = det['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
                for class_name, count in sorted(class_counts.items()):
                    print(f"  {class_name}: {count}")
        else:
            print("\n  No test images found")
    else:
        print(f"\n  Test images directory not found: {test_images_dir}")

else:
    print("\n  Skipping inference testing - no weights found")


# ==================== STEP 5: SAVE MODEL INFO ====================
print("\n" + "="*80)
print("STEP 5: SAVING MODEL INFORMATION")
print("="*80)

if best_weights and best_weights.exists():
    model_info = {
        'model_type': 'YOLOv5',
        'model_size': CONFIG['model_size'],
        'weights_path': str(best_weights),
        'dataset_yaml': str(dataset_yaml),
        'num_classes': len(CONFIG['categories']),
        'classes': CONFIG['categories'],
        'training_config': {
            'epochs': CONFIG['epochs'],
            'batch_size': CONFIG['batch_size'],
            'img_size': CONFIG['img_size'],
            'patience': CONFIG['patience']},
        'evaluation_metrics': metrics if 'metrics' in locals() else {},
        'inference_config': {
            'conf_threshold': CONFIG['conf_threshold'],
            'iou_threshold': CONFIG['iou_threshold']}
    }

    # Save model info
    model_info_path = Path('models/object_detection/model_info.json')
    model_info_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n Model information saved to: {model_info_path}")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETE!")
print("="*80)

print("\n Files and Directories:")
print(f"  Training results: models/object_detection/runs/ecommerce_object_detection/train/")
print(f"  Best weights: {best_weights if best_weights else 'Not found'}")
print(f"  Evaluation results: results/object_detection/evaluation/")
print(f"  Inference samples: results/object_detection/")
print(f"  Model info: models/object_detection/model_info.json")
print("\n Next Steps:")
print("  1. Review training results in the runs directory")
print("  2. Check evaluation metrics and visualizations")
print("  3. Test inference on your own images")
print("  4. Deploy model for production use")
print("\n Usage Examples:")
print("\n  # Detect objects in an image")
print("  from src.object_detection.detect import ObjectDetector")
print(f"  detector = ObjectDetector('{best_weights if best_weights else 'path/to/weights.pt'}')")
print("  results = detector.detect_and_visualize('your_image.jpg')")
print("\n  # Detect in video")
print("  detector.detect_video('your_video.mp4', output_path='output.mp4')")
print("\n  # Real-time webcam detection")
print("  detector.detect_webcam(camera_id=0)")
print("\n All done!")
