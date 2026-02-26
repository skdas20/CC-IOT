"""Main pipeline orchestrating the entire analysis workflow."""

import json
import logging
from pathlib import Path
from datetime import datetime

from video_processor import extract_frames_from_all_videos
from gan_enhancer import enhance_all_frames
from object_detector import detect_and_analyze
from image_analyzer import ImageAnalyzer
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.RESULTS_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DroneDrone_Analysis_Pipeline:
    """Complete drone video analysis pipeline."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
    
    def run(self):
        """Execute the complete pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Drone Video Analysis Pipeline")
        logger.info("=" * 60)
        
        try:
            # Stage 1: Extract stable frames
            logger.info("\n[Stage 1/4] Extracting stable frames from videos...")
            frames_metadata = self._stage_extract_frames()
            
            if not frames_metadata:
                logger.warning("No frames extracted. Please add videos to input_videos/")
                return self.results
            
            # Stage 2: Enhance frames with GAN
            logger.info("\n[Stage 2/4] Enhancing frames with GAN model...")
            enhanced_paths = self._stage_enhance_frames(frames_metadata)
            
            # Stage 3: Detect objects
            logger.info("\n[Stage 3/4] Detecting objects in enhanced frames...")
            detections, detection_summary = self._stage_detect_objects(enhanced_paths)
            
            # Stage 4: Analyze and generate report
            logger.info("\n[Stage 4/4] Analyzing images and generating reports...")
            final_report = self._stage_analyze_and_report(enhanced_paths, detections)
            
            # Save complete results
            self._save_results(final_report, detections)
            
            logger.info("\n" + "=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {config.RESULTS_DIR}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise
    
    def _stage_extract_frames(self):
        """Stage 1: Extract stable frames from all videos."""
        try:
            frames_metadata = extract_frames_from_all_videos(
                config.INPUT_VIDEOS_DIR,
                config.EXTRACTED_FRAMES_DIR,
                config.TARGET_NUM_FRAMES
            )
            
            self.results['stages']['frame_extraction'] = {
                'status': 'completed',
                'total_frames': len(frames_metadata),
                'frames': frames_metadata[:5] if frames_metadata else []  # Sample
            }
            
            logger.info(f"✓ Extracted {len(frames_metadata)} frames")
            return frames_metadata
        
        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            self.results['stages']['frame_extraction'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def _stage_enhance_frames(self, frames_metadata):
        """Stage 2: Enhance frames using GAN model."""
        try:
            image_paths = [f['path'] for f in frames_metadata]
            enhanced_paths = enhance_all_frames(
                frames_metadata,
                config.ENHANCED_FRAMES_DIR,
                config.GAN_UPSCALE
            )
            
            self.results['stages']['gan_enhancement'] = {
                'status': 'completed',
                'total_enhanced': len(enhanced_paths),
                'upscale_factor': config.GAN_UPSCALE
            }
            
            logger.info(f"✓ Enhanced {len(enhanced_paths)} frames")
            return enhanced_paths
        
        except Exception as e:
            logger.error(f"GAN enhancement failed: {str(e)}")
            self.results['stages']['gan_enhancement'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def _stage_detect_objects(self, image_paths):
        """Stage 3: Detect objects in enhanced frames."""
        try:
            detections, analysis = detect_and_analyze(image_paths)
            
            self.results['stages']['object_detection'] = {
                'status': 'completed',
                'total_objects': analysis['total_objects_detected'],
                'unique_classes': analysis['unique_classes'],
                'summary': analysis['objects_by_class']
            }
            
            logger.info(f"✓ Detected {analysis['total_objects_detected']} objects across {analysis['unique_classes']} classes")
            return detections, analysis
        
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            self.results['stages']['object_detection'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def _stage_analyze_and_report(self, image_paths, detections):
        """Stage 4: Analyze images and generate comprehensive reports."""
        try:
            analyzer = ImageAnalyzer()
            detailed_reports = []
            
            for idx, image_path in enumerate(image_paths):
                detection_data = detections[idx].get('objects', []) if idx < len(detections) else []
                report = analyzer.generate_comprehensive_report(image_path, detection_data)
                detailed_reports.append(report)
            
            self.results['stages']['image_analysis'] = {
                'status': 'completed',
                'total_reports': len(detailed_reports),
                'sample_report': detailed_reports[0] if detailed_reports else {}
            }
            
            logger.info(f"✓ Generated {len(detailed_reports)} comprehensive reports")
            return detailed_reports
        
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            self.results['stages']['image_analysis'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def _save_results(self, detailed_reports, detections):
        """Save all results to JSON files."""
        # Save detailed analysis reports
        reports_path = config.RESULTS_DIR / 'detailed_analysis_reports.json'
        with open(reports_path, 'w') as f:
            json.dump(detailed_reports, f, indent=2)
        logger.info(f"Saved detailed reports to {reports_path}")
        
        # Save detection results
        detections_path = config.RESULTS_DIR / 'object_detections.json'
        with open(detections_path, 'w') as f:
            json.dump(detections, f, indent=2)
        logger.info(f"Saved detections to {detections_path}")
        
        # Save pipeline summary
        summary_path = config.RESULTS_DIR / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved pipeline summary to {summary_path}")


def main():
    """Run the complete pipeline."""
    pipeline = DroneDrone_Analysis_Pipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
