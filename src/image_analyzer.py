"""Analyze images for terrain, location characteristics, and metadata."""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyze drone images for various characteristics."""
    
    @staticmethod
    def analyze_image_quality(image_path):
        """Analyze image quality metrics."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Calculate metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.std(gray)
        
        return {
            'sharpness': round(laplacian, 2),
            'brightness': round(brightness, 2),
            'contrast': round(contrast, 2),
            'resolution': f"{image.shape[1]}x{image.shape[0]}"
        }
    
    @staticmethod
    def detect_terrain_type(image_path):
        """Detect terrain type from image color and texture analysis."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze color distribution
        h_hist = np.histogram(hsv[:, :, 0], bins=180)[0]
        s_hist = np.histogram(hsv[:, :, 1], bins=256)[0]
        v_hist = np.histogram(hsv[:, :, 2], bins=256)[0]
        
        # Determine dominant terrain characteristics
        terrain_analysis = {
            'color_dominance': _analyze_color_dominance(h_hist),
            'vegetation_density': _estimate_vegetation(image),
            'water_presence': _detect_water(hsv),
            'built_structure_presence': _detect_structures(image),
            'brightness_level': 'dark' if np.mean(v_hist) < 85 else ('bright' if np.mean(v_hist) > 170 else 'moderate')
        }
        
        return terrain_analysis
    
    @staticmethod
    def extract_location_description(image_path, detections):
        """Generate location description from image analysis and detections."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Analyze different regions of the image
        top_region = image[0:h//3, :]
        middle_region = image[h//3:2*h//3, :]
        bottom_region = image[2*h//3:, :]
        
        description = {
            'image_name': Path(image_path).name,
            'dimensions': f"{w}x{h}",
            'sky_presence': _analyze_sky(top_region),
            'ground_coverage': _analyze_ground(bottom_region),
            'spatial_distribution': _analyze_spatial_layout(image),
            'detected_features': _extract_features_from_detections(detections) if detections else []
        }
        
        return description
    
    @staticmethod
    def generate_comprehensive_report(image_path, detections=None):
        """Generate a comprehensive analysis report for an image."""
        analyzer = ImageAnalyzer()
        
        quality = analyzer.analyze_image_quality(image_path)
        terrain = analyzer.detect_terrain_type(image_path)
        location = analyzer.extract_location_description(image_path, detections)
        
        report = {
            'file': Path(image_path).name,
            'quality_metrics': quality,
            'terrain_analysis': terrain,
            'location_description': location,
            'detected_objects': detections if detections else []
        }
        
        return report


def _analyze_color_dominance(h_hist):
    """Analyze dominant colors from hue histogram."""
    hue_ranges = {
        'red': (h_hist[0:15].sum() + h_hist[165:180].sum()),
        'green': h_hist[35:85].sum(),
        'blue': h_hist[100:130].sum(),
        'yellow': h_hist[15:35].sum(),
        'cyan': h_hist[85:100].sum()
    }
    dominant = max(hue_ranges, key=hue_ranges.get)
    return dominant


def _estimate_vegetation(image):
    """Estimate vegetation density from green color."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green color range in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_percentage = (cv2.countNonZero(green_mask) / green_mask.size) * 100
    
    if green_percentage > 30:
        return 'high'
    elif green_percentage > 15:
        return 'medium'
    else:
        return 'low'


def _detect_water(hsv):
    """Detect water presence in image."""
    # Blue/cyan color range in HSV
    lower_water = np.array([85, 50, 50])
    upper_water = np.array([130, 255, 255])
    
    water_mask = cv2.inRange(hsv, lower_water, upper_water)
    water_percentage = (cv2.countNonZero(water_mask) / water_mask.size) * 100
    
    return 'yes' if water_percentage > 10 else 'no'


def _detect_structures(image):
    """Detect built structures and artificial features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    large_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    return 'yes' if len(large_contours) > 10 else 'no'


def _analyze_sky(region):
    """Analyze sky presence in top region."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    sky_brightness = np.mean(gray)
    return 'clear' if sky_brightness > 150 else ('cloudy' if sky_brightness > 100 else 'dark')


def _analyze_ground(region):
    """Analyze ground/bottom region characteristics."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return 'visible' if np.std(gray) > 30 else 'obscured'


def _analyze_spatial_layout(image):
    """Analyze spatial distribution of objects."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Divide into 9 regions and check density
    regions = []
    for i in range(3):
        for j in range(3):
            region = gray[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            regions.append(np.mean(region))
    
    max_density = max(regions)
    max_idx = regions.index(max_density)
    
    positions = ['top-left', 'top-center', 'top-right',
                'mid-left', 'mid-center', 'mid-right',
                'bottom-left', 'bottom-center', 'bottom-right']
    
    return positions[max_idx]


def _extract_features_from_detections(detections):
    """Extract feature list from detection data."""
    if not detections or not isinstance(detections, list):
        return []
    
    features = []
    for detection in detections:
        if isinstance(detection, dict) and 'class' in detection:
            features.append(detection['class'])
    
    return list(set(features))  # Remove duplicates
