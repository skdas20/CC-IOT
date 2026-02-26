# Drone Video Analysis Pipeline - IoT & Cloud Computing Project

## Overview

A complete end-to-end pipeline for analyzing drone videos that:
1. **Extracts Stable Frames** - Intelligently extracts 30-40 high-quality, diverse frames from drone videos
2. **Enhances Images with GAN** - Upscales and enhances image quality using advanced GAN models
3. **Detects Objects** - Performs real-time object detection using YOLOv8
4. **Analyzes & Reports** - Identifies terrain types, location characteristics, and generates comprehensive reports

## Features

### 🎬 Video Processing
- Automatic frame extraction from drone videos
- Stability detection using histogram comparison and blur analysis
- Selective frame sampling to capture diverse viewpoints
- Support for multiple video formats (MP4, AVI, MOV, MKV, FLV, WMV)

### 🎨 GAN-Based Image Enhancement
- Super-resolution using Real-ESRGAN or fallback interpolation methods
- Automatic upscaling by 2x factor
- Sharpening and quality improvement
- GPU acceleration support

### 🔍 Object Detection
- YOLOv8 nano model for lightweight inference
- Detection of 77+ COCO object classes
- Confidence thresholding and NMS filtering
- Batch processing with progress tracking

### 📊 Comprehensive Analysis
- Image quality metrics (sharpness, brightness, contrast)
- Terrain type detection (vegetation, water, structures)
- Sky and ground coverage analysis
- Spatial layout and composition analysis
- Automatic report generation in JSON format

## Project Structure

```
CC-IOT/
├── src/
│   ├── config.py              # Configuration settings
│   ├── video_processor.py     # Video frame extraction
│   ├── gan_enhancer.py        # GAN-based image enhancement
│   ├── object_detector.py     # Object detection with YOLOv8
│   ├── image_analyzer.py      # Image analysis and reporting
│   └── pipeline.py            # Main orchestration pipeline
├── input_videos/              # Place your drone videos here
├── output/
│   ├── extracted_frames/      # Raw extracted frames
│   ├── enhanced_frames/       # GAN-enhanced frames
│   └── results/
│       ├── detailed_analysis_reports.json
│       ├── object_detections.json
│       ├── pipeline_summary.json
│       └── pipeline.log
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster processing

### Setup Steps

1. **Clone or navigate to the project:**
```bash
cd CC-IOT
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models (optional but recommended):**
```bash
# YOLOv8 will auto-download on first run
# For Real-ESRGAN, it will download on first enhancement
```

## Usage

### 1. Prepare Input Videos

Place your drone videos in the `input_videos/` directory:
```bash
cp /path/to/your/video.mp4 input_videos/
```

Supported formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- FLV (.flv)
- WMV (.wmv)

### 2. Run the Pipeline

```bash
python main.py
```

The pipeline will:
1. Extract stable frames from all videos
2. Enhance frames using GAN
3. Detect objects in enhanced frames
4. Generate comprehensive analysis reports

### 3. View Results

Results are saved in the `output/` directory:

```
output/
├── extracted_frames/          # 30-40 raw frames per video
├── enhanced_frames/           # GAN-enhanced versions
└── results/
    ├── detailed_analysis_reports.json     # Per-frame analysis
    ├── object_detections.json             # All detected objects
    ├── pipeline_summary.json              # Pipeline execution summary
    └── pipeline.log                       # Complete execution log
```

## Configuration

Edit `src/config.py` to customize:

```python
# Number of frames to extract
TARGET_NUM_FRAMES = 35

# Stability threshold (0-1, higher = more strict)
FRAME_STABILITY_THRESHOLD = 0.85

# Blur detection threshold
BLUE_THRESHOLD = 50

# GAN upscaling factor
GAN_UPSCALE = 2

# Object detection confidence
CONFIDENCE_THRESHOLD = 0.4

# Use CUDA (GPU) if available
USE_CUDA = True
```

## Output Files

### 1. Detailed Analysis Reports (`detailed_analysis_reports.json`)
Per-frame analysis including:
- Image quality metrics (sharpness, brightness, contrast)
- Terrain analysis (vegetation density, water presence, structures)
- Sky/ground coverage assessment
- Spatial distribution of objects
- Detected features

### 2. Object Detections (`object_detections.json`)
Complete detection results:
- Bounding box coordinates
- Class names
- Confidence scores
- Summary statistics by object class

### 3. Pipeline Summary (`pipeline_summary.json`)
Execution overview:
- Timestamp
- Processing stages and status
- Statistics for each stage
- Sample outputs

## Processing Pipeline Flow

```
Input Videos
    ↓
[Stage 1] Frame Extraction
    • Analyze video frames
    • Detect blurry frames
    • Select stable, diverse frames (30-40)
    ↓
[Stage 2] GAN Enhancement
    • Upscale images 2x
    • Improve quality
    • Enhance details
    ↓
[Stage 3] Object Detection
    • Run YOLOv8 inference
    • Extract bounding boxes
    • Calculate confidence scores
    ↓
[Stage 4] Analysis & Reporting
    • Terrain type detection
    • Quality metrics
    • Location characteristics
    • Generate JSON reports
    ↓
Output Results
    • Frames (extracted & enhanced)
    • Analysis reports
    • Detection results
    • Summary statistics
```

## Performance Tips

1. **Use GPU:** Install CUDA-enabled PyTorch for 5-10x faster processing
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Use Smaller Videos:** Test with shorter videos first to validate setup

3. **Adjust Parameters:** 
   - Lower `FRAME_STABILITY_THRESHOLD` to extract more frames
   - Increase `SKIP_FRAMES` to get more diverse views
   - Adjust `CONFIDENCE_THRESHOLD` for stricter/looser detection

4. **Parallel Processing:** Videos are processed sequentially; for batch jobs, run multiple commands in parallel terminals

## Troubleshooting

### No videos found error
```
Place video files in the input_videos/ directory
Supported: .mp4, .avi, .mov, .mkv, .flv, .wmv
```

### OpenCV issues
```bash
pip install --upgrade opencv-python
```

### CUDA/GPU not found
```bash
# Falls back to CPU automatically
# For GPU support, install CUDA-enabled PyTorch
```

### Out of memory
```
Reduce GAN_UPSCALE or process videos separately
```

## IOT & Cloud Computing Integration

This project demonstrates:
- **IoT Data Source:** Drone video feeds (sensor data collection)
- **Edge Processing:** Video processing on local machine
- **Cloud Inference:** Machine learning models (GAN, YOLO)
- **Data Analysis:** Terrain and location analysis
- **Result Storage:** JSON-based data format for cloud storage/API

## Example Output

```json
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
    "built_structure_presence": "no",
    "brightness_level": "bright"
  },
  "location_description": {
    "image_name": "frame_001_enhanced.jpg",
    "dimensions": "1920x1080",
    "sky_presence": "clear",
    "ground_coverage": "visible",
    "spatial_distribution": "mid-center",
    "detected_features": ["person", "car", "tree"]
  },
  "detected_objects": [
    {
      "id": 0,
      "class": "person",
      "confidence": 0.89,
      "bbox": {
        "x1": 400.5,
        "y1": 200.3,
        "x2": 500.8,
        "y2": 600.2,
        "width": 100.3,
        "height": 399.9
      }
    }
  ]
}
```

## Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB+ recommended)
- **Storage:** 2GB minimum for models and results
- **GPU:** (Optional) NVIDIA GPU with 2GB+ VRAM for acceleration

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions, check the pipeline.log in the output/results directory for detailed error messages.

---

**Happy analyzing! 🚁📊**