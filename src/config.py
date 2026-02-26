"""Configuration settings for the drone video analysis pipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_VIDEOS_DIR = PROJECT_ROOT / "input_videos"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXTRACTED_FRAMES_DIR = OUTPUT_DIR / "extracted_frames"
ENHANCED_FRAMES_DIR = OUTPUT_DIR / "enhanced_frames"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories if they don't exist
EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
ENHANCED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Video processing parameters
TARGET_NUM_FRAMES = 35  # Extract 30-40 stable frames
FRAME_STABILITY_THRESHOLD = 0.92  # Similarity threshold - frames MORE similar than this are skipped (too similar)
MIN_SCENE_CHANGE = 0.15  # Minimum difference to consider it a new view
SKIP_FRAMES = 15  # Skip N frames between samples for diversity
BLUR_THRESHOLD = 1.0  # Laplacian variance threshold (drone footage has low values ~1-25)

# GAN Enhancement parameters
GAN_MODEL = "RealESRGAN"  # Model type
GAN_UPSCALE = 4  # Upscaling factor (Real-ESRGAN x4)
USE_CUDA = True  # Use GPU if available

# Object Detection parameters
CONFIDENCE_THRESHOLD = 0.15  # Low threshold for aerial/drone footage
NMS_THRESHOLD = 0.45
DETECTION_MODEL = "yolov8m"  # Medium model - better for aerial views
DETECTION_IMGSZ = 1280  # High resolution inference for small objects

# Output parameters
SAVE_INTERMEDIATE = True  # Save extracted and enhanced frames
OUTPUT_FORMAT = "json"  # JSON output for analysis results
