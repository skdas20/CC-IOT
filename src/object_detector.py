"""Object detection on enhanced frames using YOLOv8."""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)

# COCO classes for object detection
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "cat", 15: "dog", 16: "horse", 17: "sheep", 18: "cow", 19: "elephant",
    20: "bear", 21: "zebra", 22: "giraffe", 23: "backpack", 24: "umbrella",
    25: "handbag", 26: "tie", 27: "suitcase", 28: "frisbee", 29: "skis",
    30: "snowboard", 31: "sports ball", 32: "kite", 33: "baseball bat",
    34: "baseball glove", 35: "skateboard", 36: "surfboard", 37: "tennis racket",
    38: "bottle", 39: "wine glass", 40: "cup", 41: "fork", 42: "knife",
    43: "spoon", 44: "bowl", 45: "banana", 46: "apple", 47: "sandwich",
    48: "orange", 49: "broccoli", 50: "carrot", 51: "hot dog", 52: "pizza",
    53: "donut", 54: "cake", 55: "chair", 56: "couch", 57: "potted plant",
    58: "bed", 59: "dining table", 60: "toilet", 61: "tv", 62: "laptop",
    63: "mouse", 64: "remote", 65: "keyboard", 66: "microwave", 67: "oven",
    68: "toaster", 69: "sink", 70: "refrigerator", 71: "book", 72: "clock",
    73: "vase", 74: "scissors", 75: "teddy bear", 76: "hair drier",
    77: "toothbrush"
}


class ObjectDetector:
    """Detect objects in images using YOLOv8.
    
    Uses yolov8m (medium) model with high-resolution inference (1280px)
    optimized for aerial/drone footage where objects appear small.
    """
    
    def __init__(self, confidence_threshold=0.15, nms_threshold=0.45, imgsz=1280):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.imgsz = imgsz  # High res inference for aerial/drone shots
        self.model = self._load_model()
    
    def _load_model(self):
        """Load YOLOv8-medium model (better accuracy for aerial views)."""
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8-medium model for aerial object detection...")
            model = YOLO('yolov8m.pt')  # medium model - much better for small objects
            return model
        except ImportError:
            logger.warning("YOLOv8 not available, installing...")
            import subprocess
            subprocess.run(['pip', 'install', '-q', 'ultralytics'], check=True)
            from ultralytics import YOLO
            model = YOLO('yolov8m.pt')
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            return None
    
    def detect_objects(self, image_path):
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detected objects and their properties
        """
        if self.model is None:
            return {'error': 'Model not loaded', 'objects': []}
        
        try:
            # Run detection with high resolution for aerial/drone footage
            results = self.model(str(image_path), conf=self.confidence_threshold, 
                               iou=self.nms_threshold, imgsz=self.imgsz, verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        
                        class_name = COCO_CLASSES.get(class_id, f"Unknown_{class_id}")
                        
                        detections.append({
                            'id': i,
                            'class': class_name,
                            'confidence': round(confidence, 3),
                            'bbox': {
                                'x1': round(x1, 2),
                                'y1': round(y1, 2),
                                'x2': round(x2, 2),
                                'y2': round(y2, 2),
                                'width': round(x2 - x1, 2),
                                'height': round(y2 - y1, 2)
                            }
                        })
            
            return {
                'image': str(image_path),
                'total_objects': len(detections),
                'objects': detections
            }
        
        except Exception as e:
            logger.error(f"Error detecting objects in {image_path}: {str(e)}")
            return {'error': str(e), 'objects': []}
    
    def detect_batch(self, image_paths):
        """Detect objects in multiple images."""
        detections_list = []
        
        for image_path in tqdm(image_paths, desc="Detecting objects"):
            detection = self.detect_objects(image_path)
            detections_list.append(detection)
        
        return detections_list


def analyze_detections(detections_list):
    """
    Analyze detected objects across all frames.
    
    Returns:
        Summary of object detections and object statistics
    """
    object_summary = {}
    total_detections = 0
    
    for detection in detections_list:
        if 'objects' in detection:
            for obj in detection['objects']:
                class_name = obj['class']
                total_detections += 1
                
                if class_name not in object_summary:
                    object_summary[class_name] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'appearances': []
                    }
                
                object_summary[class_name]['count'] += 1
                object_summary[class_name]['appearances'].append(obj['confidence'])
    
    # Calculate statistics
    for obj_class, stats in object_summary.items():
        if stats['appearances']:
            stats['avg_confidence'] = round(
                sum(stats['appearances']) / len(stats['appearances']), 3
            )
            stats.pop('appearances')
    
    return {
        'total_objects_detected': total_detections,
        'unique_classes': len(object_summary),
        'objects_by_class': object_summary,
        'total_frames_analyzed': len(detections_list)
    }


def detect_and_analyze(image_paths):
    """
    Run full detection and analysis pipeline.
    
    Args:
        image_paths: List of image file paths to analyze
        
    Returns:
        Tuple of (detections_list, analysis_summary)
    """
    detector = ObjectDetector()
    detections = detector.detect_batch(image_paths)
    analysis = analyze_detections(detections)
    
    logger.info(f"Detected {analysis['total_objects_detected']} objects")
    logger.info(f"Unique classes: {analysis['unique_classes']}")
    logger.info(f"Objects by class: {analysis['objects_by_class']}")
    
    return detections, analysis
