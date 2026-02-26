#!/usr/bin/env python3
"""
Main entry point for the Drone Video Analysis Pipeline.

This script demonstrates the IoT & Cloud Computing project that:
1. Extracts stable frames from drone videos
2. Enhances image quality using GAN model
3. Performs object detection on enhanced images
4. Analyzes terrain, location characteristics, and provides comprehensive reports
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline import DroneDrone_Analysis_Pipeline
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if required directories exist and have content."""
    if not config.INPUT_VIDEOS_DIR.exists():
        logger.warning(f"Videos directory not found: {config.INPUT_VIDEOS_DIR}")
        return False
    
    video_files = list(config.INPUT_VIDEOS_DIR.glob("*.*"))
    if not video_files:
        logger.warning(f"No video files found in {config.INPUT_VIDEOS_DIR}")
        logger.info("Please add your drone videos to the input_videos/ directory")
        return False
    
    logger.info(f"Found {len(video_files)} video file(s)")
    return True


def main():
    """Execute the drone analysis pipeline."""
    print("\n" + "=" * 70)
    print("  DRONE VIDEO ANALYSIS PIPELINE - IoT & Cloud Computing Project")
    print("=" * 70 + "\n")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Cannot proceed without input videos.")
        print("\nHow to use:")
        print("1. Place your drone videos in the 'input_videos/' directory")
        print("2. Run this script again")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv\n")
        return 1
    
    try:
        # Run the pipeline
        pipeline = DroneDrone_Analysis_Pipeline()
        results = pipeline.run()
        
        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        
        for stage, details in results.get('stages', {}).items():
            status = details.get('status', 'unknown')
            icon = "✓" if status == 'completed' else "✗"
            print(f"{icon} {stage}: {status}")
            
            if status == 'completed':
                if 'total_frames' in details:
                    print(f"  → Extracted {details['total_frames']} frames")
                if 'total_enhanced' in details:
                    print(f"  → Enhanced {details['total_enhanced']} frames")
                if 'total_objects' in details:
                    print(f"  → Detected {details['total_objects']} objects ({details['unique_classes']} classes)")
                if 'total_reports' in details:
                    print(f"  → Generated {details['total_reports']} analysis reports")
        
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print("\nOutput files:")
        print(f"  • {config.RESULTS_DIR}/detailed_analysis_reports.json")
        print(f"  • {config.RESULTS_DIR}/object_detections.json")
        print(f"  • {config.RESULTS_DIR}/pipeline_summary.json")
        print(f"  • {config.EXTRACTED_FRAMES_DIR}/ (extracted frames)")
        print(f"  • {config.ENHANCED_FRAMES_DIR}/ (enhanced frames)")
        
        print("\n" + "=" * 70 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\n✗ Error: {str(e)}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
