"""Extract stable frames from drone videos."""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process drone videos to extract stable frames."""
    
    def __init__(self, max_similarity=0.92, min_scene_change=0.15,
                 skip_frames=15, blur_threshold=1.0):
        self.max_similarity = max_similarity  # Skip if TOO similar to prev (duplicate)
        self.min_scene_change = min_scene_change  # Min difference for new view
        self.skip_frames = skip_frames
        self.blur_threshold = blur_threshold  # Drone footage has low Laplacian values
        
    def calculate_laplacian_variance(self, frame):
        """Calculate Laplacian variance to detect blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_frame_difference(self, frame1, frame2):
        """Calculate difference between two frames using multiple metrics."""
        # Method 1: Histogram comparison (Bhattacharyya distance)
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [16, 16, 16], 
                            [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [16, 16, 16], 
                            [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        
        # Method 2: Structural difference (mean absolute difference)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Resize to same size for comparison
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        g1 = cv2.resize(gray1, (w, h))
        g2 = cv2.resize(gray2, (w, h))
        structural_diff = np.mean(np.abs(g1.astype(float) - g2.astype(float))) / 255.0
        
        # Combined score: higher = more different
        combined = (hist_diff * 0.6) + (structural_diff * 0.4)
        return combined
    
    def is_good_frame(self, frame, prev_frame=None):
        """
        Check if frame is good to extract:
        1. Not too blurry
        2. Sufficiently different from previous selected frame
        3. Not too dark or washed out
        """
        # Check blur
        blur_score = self.calculate_laplacian_variance(frame)
        if blur_score < self.blur_threshold:
            return False, blur_score, 'too_blurry'
        
        # Check brightness (skip very dark or very bright frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 15 or mean_brightness > 245:
            return False, blur_score, 'bad_exposure'
        
        # Check difference from previous frame
        if prev_frame is not None:
            diff = self.calculate_frame_difference(frame, prev_frame)
            similarity = 1.0 - diff
            
            # Skip if too similar to previous (near-duplicate)
            if similarity > self.max_similarity:
                return False, blur_score, 'too_similar'
            
            return True, blur_score, f'diff={diff:.3f}'
        
        # First frame is always accepted
        return True, blur_score, 'first_frame'
    
    def extract_stable_frames(self, video_path, target_num_frames=35):
        """
        Extract stable, diverse frames from video.
        Uses adaptive sampling: distributes frames evenly across video duration.
        
        Returns:
            list: List of tuples (frame, frame_number, blur_score)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {Path(video_path).name}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.1f}")
        
        if total_frames < 10:
            cap.release()
            logger.warning(f"Video too short: {total_frames} frames")
            return []
        
        # Calculate adaptive skip interval to spread frames evenly
        # We want ~target_num_frames from the whole video
        adaptive_skip = max(self.skip_frames, total_frames // (target_num_frames * 3))
        
        stable_frames = []
        prev_selected_frame = None
        frame_count = 0
        
        with tqdm(total=total_frames, desc=f"Extracting [{Path(video_path).stem}]") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                pbar.update(1)
                
                # Skip frames for diversity
                if frame_count % adaptive_skip != 0:
                    continue
                
                # Check if this is a good frame
                is_good, blur_score, reason = self.is_good_frame(frame, prev_selected_frame)
                
                if is_good:
                    stable_frames.append((frame.copy(), frame_count, blur_score))
                    prev_selected_frame = frame.copy()
                
                # Stop when we have enough frames
                if len(stable_frames) >= target_num_frames:
                    break
        
        cap.release()
        
        logger.info(f"Extracted {len(stable_frames)} stable frames from {Path(video_path).name} "
                   f"(adaptive_skip={adaptive_skip})")
        
        return stable_frames
    
    def save_extracted_frames(self, frames, output_dir):
        """Save extracted frames to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        for idx, (frame, frame_num, score) in enumerate(frames):
            filename = f"frame_{idx:03d}_raw.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            frame_paths.append({
                'index': idx,
                'frame_number': frame_num,
                'stability_score': float(score),
                'path': str(filepath)
            })
        
        return frame_paths


def extract_frames_from_all_videos(input_dir, output_dir, target_num_frames=35):
    """Process all videos in a directory, distributing target frames across videos."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    # Deduplicate
    video_files = list(set(video_files))
    
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Distribute target frames across videos
    frames_per_video = max(5, target_num_frames // len(video_files))
    logger.info(f"Extracting ~{frames_per_video} frames per video (target total: {target_num_frames})")
    
    processor = VideoProcessor()
    all_frames_metadata = []
    
    for video_file in sorted(video_files):
        try:
            frames = processor.extract_stable_frames(video_file, frames_per_video)
            if frames:
                metadata = processor.save_extracted_frames(
                    frames, 
                    output_dir / video_file.stem
                )
                all_frames_metadata.extend(metadata)
            else:
                logger.warning(f"No frames extracted from {video_file.name}")
        except Exception as e:
            logger.error(f"Error processing {video_file}: {str(e)}", exc_info=True)
    
    logger.info(f"Total frames extracted across all videos: {len(all_frames_metadata)}")
    return all_frames_metadata
