"""Configuration settings for the drone video analysis pipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_VIDEOS_DIR = PROJECT_ROOT / "input_videos"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXTRACTED_FRAMES_DIR = OUTPUT_DIR / "extracted_frames"
ENHANCED_FRAMES_DIR = OUTPUT_DIR / "enhanced_frames"
DEPTH_MAPS_DIR = OUTPUT_DIR / "depth_maps"
SCENE_MAPS_DIR = OUTPUT_DIR / "scene_maps"
RESULTS_DIR = OUTPUT_DIR / "results"
LOCAL_MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "cache"

# Create directories if they don't exist
EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
ENHANCED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_MAPS_DIR.mkdir(parents=True, exist_ok=True)
SCENE_MAPS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Video processing parameters
TARGET_NUM_FRAMES = 7   # 1 representative frame per video (faster demo, cleaner story)
FRAME_STABILITY_THRESHOLD = 0.92  # Similarity threshold - frames MORE similar than this are skipped (too similar)
MIN_SCENE_CHANGE = 0.15  # Minimum difference to consider it a new view
SKIP_FRAMES = 15  # Skip N frames between samples for diversity
BLUR_THRESHOLD = 1.0  # Laplacian variance threshold (drone footage has low values ~1-25)

# GAN Enhancement parameters
GAN_MODEL = "RealESRGAN"  # Model type
GAN_UPSCALE = 4  # Upscaling factor (Real-ESRGAN x4)
USE_CUDA = True  # Use GPU if available
GAN_MAX_INPUT_DIM_CPU = 256  # Smaller tiles → ~5x faster on CPU

# Object Detection parameters
CONFIDENCE_THRESHOLD = 0.35  # Raised from 0.15 - filters low-confidence aerial mismatches
NMS_THRESHOLD = 0.45
DETECTION_MODEL = "yolov8m"  # Medium model - better for aerial views
DETECTION_IMGSZ = 640   # Reduced for CPU speed

# Aerial-relevant COCO class IDs only (50-60m drone height context)
# Excludes indoor objects, food, furniture, sports gear — impossible at this altitude
AERIAL_CLASS_IDS = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    7,   # truck
    8,   # boat
    9,   # traffic light
    11,  # stop sign
    13,  # bench
    14,  # bird
    15,  # dog (rare but possible)
]

# Depth estimation parameters
DEPTH_MODEL = "DPT_Hybrid"  # Already cached locally, good presentation quality

# Output parameters
SAVE_INTERMEDIATE = True  # Save extracted and enhanced frames
OUTPUT_FORMAT = "json"  # JSON output for analysis results
