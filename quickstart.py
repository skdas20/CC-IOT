#!/usr/bin/env python3
"""
QUICK START GUIDE - Drone Video Analysis Pipeline

This script demonstrates how to use the pipeline with minimal setup.
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          DRONE VIDEO ANALYSIS PIPELINE - IoT & Cloud Computing            ║
║                                                                            ║
║                        Quick Start Guide                                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

PROJECT FEATURES:
═══════════════════════════════════════════════════════════════════════════

✓ Frame Extraction       - Extract 30-40 stable frames from drone videos
✓ GAN Enhancement       - Upscale images 2x with quality improvement  
✓ Object Detection      - Detect 77+ object types using YOLOv8
✓ Terrain Analysis      - Identify vegetation, water, structures
✓ Location Description  - Sky presence, ground coverage, spatial layout
✓ Auto Reporting        - Generate JSON reports for all findings


QUICK START:
═══════════════════════════════════════════════════════════════════════════

1. PREPARE INPUT VIDEOS
   ─────────────────────
   
   Copy your drone videos to: input_videos/
   
   Supported formats:
   • MP4 (.mp4)      • AVI (.avi)      • MOV (.mov)
   • MKV (.mkv)      • FLV (.flv)      • WMV (.wmv)
   
   Example:
   $ cp ~/my_drone_video.mp4 input_videos/


2. RUN THE PIPELINE
   ─────────────────
   
   From the CC-IOT directory:
   $ python3 main.py
   
   The pipeline will:
   - Extract stable frames from all videos
   - Enhance frames using GAN model
   - Run object detection on enhanced frames
   - Generate comprehensive analysis reports


3. VIEW RESULTS
   ─────────────
   
   All results are saved in: output/
   
   📁 output/
      ├── extracted_frames/              (raw extracted frames)
      ├── enhanced_frames/               (GAN-enhanced frames)
      └── results/
          ├── detailed_analysis_reports.json
          ├── object_detections.json
          ├── pipeline_summary.json
          └── pipeline.log


ADVANCED CONFIGURATION:
═══════════════════════════════════════════════════════════════════════════

Edit src/config.py to customize:

    TARGET_NUM_FRAMES = 35              # Number of frames to extract
    FRAME_STABILITY_THRESHOLD = 0.85    # Stability sensitivity (0-1)
    GAN_UPSCALE = 2                     # Image upscaling factor
    CONFIDENCE_THRESHOLD = 0.4          # Object detection confidence
    USE_CUDA = True                     # Use GPU acceleration


SAMPLE OUTPUT:
═══════════════════════════════════════════════════════════════════════════

Detailed Analysis Report (JSON):

{
  "file": "frame_001_enhanced.jpg",
  "quality_metrics": {
    "sharpness": 285.42,
    "brightness": 128.5,
    "contrast": 45.3,
    "resolution": "1920x1080"
  },
  "terrain_analysis": {
    "color_dominance": "green",
    "vegetation_density": "high",
    "water_presence": "no",
    "built_structure_presence": "yes",
    "brightness_level": "bright"
  },
  "detected_objects": [
    {
      "class": "car",
      "confidence": 0.92,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    },
    {
      "class": "person",
      "confidence": 0.85,
      "bbox": {"x1": 400, "y1": 150, "x2": 450, "y2": 500}
    }
  ]
}


FILE STRUCTURE:
═══════════════════════════════════════════════════════════════════════════

CC-IOT/
├── main.py                         # Entry point (main script)
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
│
├── src/                            # Source code modules
│   ├── config.py                   # Configuration settings
│   ├── video_processor.py          # Video frame extraction
│   ├── gan_enhancer.py             # GAN enhancement
│   ├── object_detector.py          # YOLOv8 detection
│   ├── image_analyzer.py           # Image analysis
│   └── pipeline.py                 # Main pipeline orchestration
│
├── input_videos/                   # YOUR DRONE VIDEOS GO HERE
│   └── (place .mp4, .avi, etc.)
│
└── output/                         # Results directory
    ├── extracted_frames/           # Raw frames
    ├── enhanced_frames/            # Enhanced frames
    └── results/                    # Analysis results (JSON)


SYSTEM REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════════

Minimum:
  • Python 3.8+
  • 4 GB RAM
  • 2 GB storage

Recommended:
  • Python 3.10+
  • 8 GB RAM  
  • NVIDIA GPU (2GB+ VRAM)
  • 5 GB storage


DEPENDENCIES INSTALLED:
═══════════════════════════════════════════════════════════════════════════

✓ opencv-python     4.13.0+    (Video/image processing)
✓ numpy             2.4.2+     (Numerical computing)
✓ torch             2.0.0+     (Deep learning framework)
✓ torchvision       0.15.0+    (Vision models)
✓ ultralytics       8.0.0+     (YOLOv8 object detection)
✓ scikit-image      0.26.0+    (Image processing)
✓ matplotlib        3.10.8+    (Visualization)
✓ Pillow            12.1.1+    (Image library)
✓ scipy             1.17.1+    (Scientific computing)
✓ tqdm              4.67.3+    (Progress bars)
✓ requests          2.32.5+    (HTTP library)


NEXT STEPS:
═══════════════════════════════════════════════════════════════════════════

1. Add drone videos to input_videos/ directory
2. Run: python3 main.py
3. Check output/ directory for results
4. Review JSON reports for complete analysis


For full documentation, see: README.md

═══════════════════════════════════════════════════════════════════════════
""")

# Try to import pipeline modules to verify setup
print("\n📦 Verifying module imports...\n")

try:
    from src.config import *
    print("✓ Config module loaded")
except Exception as e:
    print(f"✗ Config module error: {e}")

try:
    from src.video_processor import VideoProcessor
    print("✓ Video processor loaded")
except Exception as e:
    print(f"✗ Video processor error: {e}")

try:
    from src.gan_enhancer import SimpleGANEnhancer
    print("✓ GAN enhancer loaded")
except Exception as e:
    print(f"✗ GAN enhancer error: {e}")

try:
    from src.object_detector import ObjectDetector
    print("✓ Object detector loaded")
except Exception as e:
    print(f"✗ Object detector error: {e}")

try:
    from src.image_analyzer import ImageAnalyzer
    print("✓ Image analyzer loaded")
except Exception as e:
    print(f"✗ Image analyzer error: {e}")

try:
    from src.pipeline import DroneDrone_Analysis_Pipeline
    print("✓ Pipeline orchestrator loaded")
except Exception as e:
    print(f"✗ Pipeline error: {e}")

print("\n✅ All modules loaded successfully!\n")
print("Ready to process drone videos. Place videos in 'input_videos/' and run:")
print("   python3 main.py\n")
