"""
Aerial scene zone segmentation using HSV color analysis.

Labels each region of a drone image as one of:
  Sky | Vegetation | Built Structure | Road/Ground | Water

No ML model needed — uses color and texture properties that are
physically reliable for aerial urban/residential imagery.
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Zone colors for the overlay (BGR)
ZONE_COLORS = {
    'sky':        (220, 180,  50),   # light blue
    'vegetation': ( 30, 160,  30),   # green
    'built':      ( 60,  90, 180),   # orange-brown (buildings/rooftops/walls)
    'road':       (100, 100, 100),   # grey (road/ground/cement)
    'water':      (180,  80,  20),   # blue-ish (water bodies/water tanks)
}

ZONE_LABELS = {
    'sky':        'Sky',
    'vegetation': 'Vegetation / Trees',
    'built':      'Built Structure / Rooftop',
    'road':       'Road / Ground / Cement',
    'water':      'Water / Water Tank',
}


def segment_scene(image_bgr):
    """
    Assign each pixel to a scene zone using HSV thresholds.
    Returns a label map (H x W) with string keys and a colored mask (H x W x 3).
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Sky: bright + low-saturation OR blue-ish -------------------------
    sky_bright = (V > 160) & (S < 70)
    sky_blue   = (H >= 95) & (H <= 135) & (S > 40) & (V > 100)
    sky_mask   = sky_bright | sky_blue

    # ---- Vegetation: green hue, reasonable saturation & brightness --------
    veg_mask = (H >= 30) & (H <= 90) & (S > 40) & (V > 30)

    # ---- Water: blue-cyan hue with medium saturation ----------------------
    water_mask = (H >= 85) & (H <= 135) & (S > 60) & (V > 40) & (~sky_mask)

    # ---- Road / bare ground: low saturation, mid-dark grey ----------------
    # Edge density distinguishes road (flat) from structures (edgy)
    edges = cv2.Canny(gray, 60, 140)
    # Dilate edges so pixels near edges count as structured
    kernel = np.ones((9, 9), np.uint8)
    edge_dense = cv2.dilate(edges, kernel).astype(bool)

    low_sat = (S < 60)
    road_mask = low_sat & (~edge_dense) & (~sky_mask) & (~veg_mask) & (~water_mask)

    # ---- Built structure: everything with edges + not green/water/sky -----
    built_mask = edge_dense & (~sky_mask) & (~veg_mask) & (~water_mask)

    # ---- Assign priority (sky > water > veg > built > road > unknown) -----
    zone_map = np.full((h, w), 'road', dtype=object)
    zone_map[built_mask]  = 'built'
    zone_map[veg_mask]    = 'vegetation'
    zone_map[water_mask]  = 'water'
    zone_map[sky_mask]    = 'sky'

    # ---- Build colored mask -----------------------------------------------
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for zone, color in ZONE_COLORS.items():
        color_mask[zone_map == zone] = color

    return zone_map, color_mask


def compute_zone_stats(zone_map):
    """Return percentage coverage per zone."""
    total = zone_map.size
    stats = {}
    for zone in ZONE_COLORS:
        pct = float(np.sum(zone_map == zone)) / total * 100
        stats[zone] = round(pct, 1)
    return stats


def create_scene_visualization(image_bgr, color_mask, zone_stats, image_name):
    """
    Produce a side-by-side image: original | blended overlay | legend.
    Returns the composite image (numpy array).
    """
    h, w = image_bgr.shape[:2]

    # Blend original + colored mask
    overlay = cv2.addWeighted(image_bgr, 0.45, color_mask, 0.55, 0)

    # ---- Legend panel ----
    legend_w = 420
    legend = np.ones((h, legend_w, 3), dtype=np.uint8) * 240  # light grey bg

    y = 40
    cv2.putText(legend, 'SCENE ZONES', (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    y += 35
    cv2.putText(legend, image_name[:40], (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
    y += 40

    for zone, label in ZONE_LABELS.items():
        pct = zone_stats.get(zone, 0)
        color = ZONE_COLORS[zone]
        # Color swatch
        cv2.rectangle(legend, (15, y - 18), (55, y + 6), color, -1)
        cv2.rectangle(legend, (15, y - 18), (55, y + 6), (50, 50, 50), 1)
        # Label + percentage
        text = f'{label}  {pct:.1f}%'
        cv2.putText(legend, text, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1)
        # Mini bar
        bar_x = 70
        bar_y = y + 12
        bar_len = int(pct / 100 * (legend_w - 90))
        cv2.rectangle(legend, (bar_x, bar_y), (bar_x + bar_len, bar_y + 7), color, -1)
        y += 65

    # Stack: original | overlay | legend
    orig_resized    = cv2.resize(image_bgr, (w, h))
    overlay_resized = cv2.resize(overlay, (w, h))
    legend_resized  = cv2.resize(legend, (legend_w, h))

    composite = np.hstack([orig_resized, overlay_resized, legend_resized])
    return composite


def run_scene_segmentation(image_paths, output_dir):
    """
    Run scene segmentation on all images and save visualizations.
    Returns list of per-image stats dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for image_path in image_paths:
        image_path = Path(image_path)
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                logger.warning(f"Cannot read: {image_path}")
                continue

            zone_map, color_mask = segment_scene(image_bgr)
            stats = compute_zone_stats(zone_map)

            composite = create_scene_visualization(
                image_bgr, color_mask, stats, image_path.stem
            )

            out_path = output_dir / f"{image_path.stem}_scene.jpg"
            cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 90])

            reports.append({
                'image': str(image_path),
                'scene_map_path': str(out_path),
                'zone_coverage_pct': stats,
                'dominant_zone': max(stats, key=stats.get),
            })
            logger.info(f"{image_path.name}: {stats}")

        except Exception as e:
            logger.error(f"Scene segmentation failed for {image_path}: {e}")

    logger.info(f"Scene maps saved: {len(reports)} images → {output_dir}")
    return reports
