"""Depth estimation utilities for drone frames."""

import os
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Estimate monocular depth maps using MiDaS."""

    def __init__(self, model_type="MiDaS_small", use_cuda=True):
        self.model_type = model_type
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load MiDaS and keep all hub assets inside the project folder."""
        torch_home = config.LOCAL_MODEL_CACHE_DIR / "torch_hub"
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)

        logger.info(f"Loading depth model {self.model_type} on {self.device}...")

        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.model_type,
            pretrained=True,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        self.transform = (
            transforms.small_transform
            if self.model_type == "MiDaS_small"
            else transforms.dpt_transform
        )

    @torch.no_grad()
    def estimate_depth(self, image_path):
        """Estimate a normalized depth map for one image."""
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image_rgb).to(self.device)

        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max - depth_min < 1e-6:
            normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            normalized = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)

        return {
            "raw_depth": depth,
            "normalized_depth": normalized,
            "stats": self._summarize_depth(normalized, image_path.name),
        }

    def _summarize_depth(self, normalized_depth, image_name):
        """Build simple summary metrics for reporting and presentation."""
        near_ratio = float(np.mean(normalized_depth >= 170))
        mid_ratio = float(np.mean((normalized_depth >= 85) & (normalized_depth < 170)))
        far_ratio = float(np.mean(normalized_depth < 85))

        return {
            "image_name": image_name,
            "min_depth_normalized": int(normalized_depth.min()),
            "max_depth_normalized": int(normalized_depth.max()),
            "mean_depth_normalized": round(float(normalized_depth.mean()), 2),
            "near_region_ratio": round(near_ratio, 3),
            "mid_region_ratio": round(mid_ratio, 3),
            "far_region_ratio": round(far_ratio, 3),
        }

    def save_depth_outputs(self, image_path, normalized_depth, output_dir):
        """Save grayscale and colorized depth maps."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        gray_path = output_dir / f"{stem}_depth_gray.png"
        color_path = output_dir / f"{stem}_depth_color.png"

        cv2.imwrite(str(gray_path), normalized_depth)
        colorized = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(color_path), colorized)

        return {
            "gray_map_path": str(gray_path),
            "color_map_path": str(color_path),
        }

    def estimate_batch(self, image_paths, output_dir):
        """Estimate depth for multiple images."""
        depth_reports = []

        for image_path in tqdm(image_paths, desc="Estimating depth"):
            try:
                result = self.estimate_depth(image_path)
                saved_paths = self.save_depth_outputs(
                    image_path, result["normalized_depth"], output_dir
                )

                depth_reports.append(
                    {
                        "image": str(image_path),
                        **saved_paths,
                        "summary": result["stats"],
                    }
                )
            except Exception as e:
                logger.error(f"Error estimating depth for {image_path}: {e}")
                depth_reports.append(
                    {
                        "image": str(image_path),
                        "error": str(e),
                    }
                )

        return depth_reports


def estimate_depth_for_images(image_paths, output_dir):
    """Run depth estimation for all enhanced images."""
    estimator = DepthEstimator(
        model_type=config.DEPTH_MODEL,
        use_cuda=config.USE_CUDA,
    )
    reports = estimator.estimate_batch(image_paths, output_dir)
    successful = [report for report in reports if "error" not in report]
    logger.info(f"Generated {len(successful)} depth maps")
    return reports
