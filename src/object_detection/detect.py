""" Inference pipeline for object detection """

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
from typing import List, Dict, Union, Tuple
from PIL import Image

class ObjectDetector:
    """ Detect objects in images using trained YOLOv5 model """
    def __init__(self,
                 weights_path: str,
                 data_yaml: str = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = None):
        """
        Initialize detector
        
        Args:
            weights_path: Path to trained weights
            data_yaml: Path to dataset YAML (for class names)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to use ('cuda' or 'cpu')
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from {self.weights_path}...")

        # Load YOLOv5 model
        try:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", 
                                       path=str(self.weights_path),
                                       device = self.device,
                                       force_reload = False)
            # Set thresholds
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold

            print(f"   Model loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   Confidence threshold: {self.conf_threshold}")
            print(f"   IoU threshold: {self.iou_threshold}")
        
        except Exception as e:
            print(f" Failed to load model: {e}")
            raise

        # Get class names
        if data_yaml and Path(data_yaml).exists():
            with open(data_yaml, "r") as f:
                data_config = yaml.safe_load(f)
            self.class_names = data_config["names"]
        else:
            self.class_names = self.model.names

        self.num_classes = len(self.class_names)
        print(f"   Number of classes: {self.num_classes}")

    def detect(self,
               image: Union[str, np.ndarray, Image.Image],
               img_size: int = 640) -> Dict:
        """
        Detect objects in image
        
        Args:
            image: Path to image, numpy array, or PIL Image
            img_size: Input image size
            
        Returns:
            Dictionary with detection results
        """
        # Load image if path provided
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        # Run inference
        results = self.model(img, size=img_size)

        # Parse results
        detections = []

        # results.pandas().xyxy[0] gives pandas DataFrame with detections
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            detection = {
                "class": row["name"],
                "class_id": int(row["class"]),
                "confidence": float(row["confidence"]),
                "bbox": [
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"])
                ]
            }
            detections.append(detection)

        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_shape": img_size
        }
    
    def detect_batch(self,
                     images: List[Union[str, np.ndarray]],
                     img_size: int = 640) -> List[Dict]:
        """
        Detect objects in multiple images
        
        Args:
            images: List of image paths or arrays
            img_size: Input image size
            
        Returns:
            List of detection results
        """
        print(f"Detecting objects in {len(images)} images...")

        results_list = []
        for img in images:
            result = self.detect(img, img_size=img_size)
            results_list.append(result)

        return results_list
    
    def visualize_detection(self,
                            image: Union[str, np.ndarray, Image.Image],
                            detections: Dict = None,
                            save_path: str = None,
                            show_conf: bool = True,
                            line_width: int = 2):
        """
        Visualize detections on image
        
        Args:
            image: Image to visualize
            detections: Detection results (if None, will run detection)
            save_path: Path to save visualization
            show_conf: Whether to show confidence scores
            line_width: Bounding box line width
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.Color_BGR2BGR)
        elif isinstance(image, Image.Image):
            img = np.ndarray(image)
        else:
            img = image.copy()

        # Get detections if not provided
        if detections is None:
            detections = self.detect(image)

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12,8))
        ax.imshow(img)

        # Generate colors for each class
        np.random.seed(42)
        colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes))
        
        # Draw detections
        for det in detections['detections']:
            bbox = det['bbox']
            class_name = det['class']
            class_id = det['class_id']
            confidence = det['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=line_width,
                edgecolor=colors[class_id],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name}"
            if show_conf:
                label += f" {confidence:.2f}"
            
            ax.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(
                    facecolor=colors[class_id],
                    alpha=0.7,
                    edgecolor='none',
                    pad=2
                )
            )
        
        ax.axis('off')
        plt.title(f"Detected {detections['num_detections']} objects", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Visualization saved to {save_path}")
        
        plt.show()
        plt.close()

    def detect_and_visualize(self,
                             image: Union[str, np.ndarray],
                             save_path: str = None) -> Dict:
        """
        Detect objects and visualize in one step
        
        Args:
            image: Image to process
            save_path: Path to save visualization
            
        Returns:
            Detection results
        """
        # Detect
        detections = self.detect(image)

        # Print results
        print(f"\nDetected {detections['num_detections']} objects:")
        for i, det in enumerate(detections['detections'], 1):
            print(f"  {i}. {det['class']} (confidence: {det['confidence']:.2f})")
        
        # Visualize
        self.visualize_detection(image, detections, save_path)

        return detections
    
    def detect_video(self,
                     video_path: str,
                     output_path: str = None,
                     display: bool = True,
                     fps: int = None):
        """
        Detect objects in video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            display: Whether to display video while processing
            fps: Output video FPS (if None, uses input FPS)
        """
        print(f"\nProcessing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f" Could not open video: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None:
            fps = input_fps

        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {input_fps}")
        print(f"  Total frames: {total_frames}")

        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Convert BGR to RGB for model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR)

                # Detect objects
                results = self.model(frame_rgb)

                # Render results on frame
                frame_with_boxes = np.squeeze(results.render())

                # Convert back to BGR for OpenCV
                frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
                
                # Write frame
                if output_path:
                    out.write(frame_with_boxes)
                
                # Display
                if display:
                    cv2.imshow('Object Detection', frame_with_boxes)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_count % 30 == 0:
                    print(f"  Processed {frame_count}/{total_frames} frames", end='\r')

        finally:
            cap.release()
            if output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"\n Video processing complete!")
        if output_path:
            print(f"   Output saved to: {output_path}")


    def detect_webcam(self, camera_id: int = 0):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID
        """
        print(f"\nStarting webcam detection (camera {camera_id})")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f" Could not open camera {camera_id}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect
                results = self.model(frame_rgb)
                
                # Render
                frame_with_boxes = np.squeeze(results.render())
                frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
                
                # Display
                cv2.imshow('Webcam Object Detection', frame_with_boxes)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print("\n Webcam detection stopped")

    
def detect_objects(weights_path: str,
                   image_path: str,
                   data_yaml: str = None,
                   save_path: str = None):
    """
    Convenience function to detect objects in image
    
    Args:
        weights_path: Path to model weights
        image_path: Path to image
        data_yaml: Path to dataset YAML
        save_path: Path to save visualization
        
    Returns:
        Detection results
    """
    detector = ObjectDetector(weights_path, data_yaml)
    detections = detector.detect_and_visualize(image_path, save_path)
    
    return detections

if __name__ == "__main__":
    print("="*80)
    print("OBJECT DETECTION - INFERENCE EXAMPLE")
    print("="*80)

    weights_path = "models/object_detection/runs/object_detection/train/weights/best.pt"
    data_yaml = "data/object_detection/dataset.yaml"

    if Path(weights_path).exists():
        detector = ObjectDetector(weights_path, data_yaml)

        # Example 1: Detect in image
        test_images_dir = Path("data/object_detection/images/test")

        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg"))[:5]

            for i, img_path in enumerate(test_images):
                print(f"\n{'='*80}")
                print(f"Processing image {i+1}/{len(test_images)}")
                print(f"{'='*80}")
                
                detections = detector.detect_and_visualize(
                    str(img_path),
                    save_path=f'results/object_detection/detection_example_{i+1}.png'
                )

        # Example 2: Webcam detection (uncomment to use)
        # detector.detect_webcam(camera_id=0)

    else:
        print(f"Weights not found: {weights_path}")
        print("Train a model first using scripts/train_object_detector.py")

