#!/usr/bin/env python3
"""
Run only Stage 3 (object detection), Stage 4 (depth estimation), and Stage 5 (analysis).
Enhanced frames already exist — skip the slow extraction + enhancement stages.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.RESULTS_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  DRONE PIPELINE — Stages 3-5 (detection + depth + analysis)")
    print("=" * 70 + "\n")

    # Collect already-enhanced frames
    enhanced_paths = sorted(
        str(p) for p in config.ENHANCED_FRAMES_DIR.glob("*.jpg")
    )
    if not enhanced_paths:
        print("ERROR: No enhanced frames found in", config.ENHANCED_FRAMES_DIR)
        return 1

    logger.info(f"Found {len(enhanced_paths)} enhanced frames")

    results = {'timestamp': datetime.now().isoformat(), 'stages': {}}

    # ------------------------------------------------------------------ #
    # Stage 3: Object Detection
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 3/3] Object detection on {len(enhanced_paths)} frames...")
    from object_detector import detect_and_analyze
    detections, detection_summary = detect_and_analyze(enhanced_paths)
    results['stages']['object_detection'] = {
        'status': 'completed',
        'total_objects': detection_summary['total_objects_detected'],
        'unique_classes': detection_summary['unique_classes'],
        'summary': detection_summary['objects_by_class'],
    }
    logger.info(f"Detected {detection_summary['total_objects_detected']} objects "
                f"across {detection_summary['unique_classes']} classes")

    # ------------------------------------------------------------------ #
    # Stage 4: Depth Estimation
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 4/3] Depth estimation on {len(enhanced_paths)} frames...")
    from depth_estimator import estimate_depth_for_images
    depth_reports = estimate_depth_for_images(enhanced_paths, config.DEPTH_MAPS_DIR)
    successful_depth = [r for r in depth_reports if 'error' not in r]
    results['stages']['depth_estimation'] = {
        'status': 'completed',
        'total_depth_maps': len(successful_depth),
    }
    logger.info(f"Generated {len(successful_depth)} depth maps")

    # ------------------------------------------------------------------ #
    # Stage 5: Image Analysis & Report
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 5/3] Generating comprehensive reports...")
    from image_analyzer import ImageAnalyzer
    analyzer = ImageAnalyzer()
    depth_by_image = {
        r['image']: r for r in depth_reports
        if isinstance(r, dict) and 'image' in r and 'error' not in r
    }

    detailed_reports = []
    for idx, image_path in enumerate(enhanced_paths):
        detection_data = detections[idx].get('objects', []) if idx < len(detections) else []
        report = analyzer.generate_comprehensive_report(image_path, detection_data)
        depth_report = depth_by_image.get(image_path)
        if depth_report:
            report['depth_analysis'] = depth_report.get('summary', {})
            report['depth_maps'] = {
                'gray_map_path': depth_report.get('gray_map_path'),
                'color_map_path': depth_report.get('color_map_path'),
            }
        detailed_reports.append(report)

    results['stages']['image_analysis'] = {
        'status': 'completed',
        'total_reports': len(detailed_reports),
    }

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    (config.RESULTS_DIR / 'detailed_analysis_reports.json').write_text(
        json.dumps(detailed_reports, indent=2))
    (config.RESULTS_DIR / 'object_detections.json').write_text(
        json.dumps(detections, indent=2))
    (config.RESULTS_DIR / 'depth_analysis.json').write_text(
        json.dumps(depth_reports, indent=2))
    (config.RESULTS_DIR / 'pipeline_summary.json').write_text(
        json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETED")
    print("=" * 70)
    for stage, info in results['stages'].items():
        print(f"  {stage}: {info['status']}")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Depth maps saved to: {config.DEPTH_MAPS_DIR}")
    print("=" * 70 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
