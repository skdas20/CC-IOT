"""GAN-based image enhancement using Real-ESRGAN for super-resolution."""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import config

logger = logging.getLogger(__name__)

# ============================================================================
# Real-ESRGAN Architecture: RRDB (Residual-in-Residual Dense Block) Network
# Same architecture as the official Real-ESRGAN project by xinntao
# ============================================================================

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB Network architecture for Real-ESRGAN (4x upscale)."""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # 4x upsample (2x + 2x)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RealESRGANEnhancer:
    """Real-ESRGAN based image super-resolution enhancer.
    
    Downloads and uses the official RealESRGAN_x4plus pretrained model
    with the RRDB architecture for 4x image upscaling.
    """
    
    MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    MODEL_FILENAME = "RealESRGAN_x4plus.pth"
    
    def __init__(self, scale=4, use_cuda=True, tile_size=192):
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = 10
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"Real-ESRGAN enhancement | device: {self.device} | scale: {self.scale}x")
        
        self.model = self._load_model()
    
    def _download_model(self, model_path):
        """Download the Real-ESRGAN pretrained weights."""
        logger.info(f"Downloading Real-ESRGAN x4plus model (~67MB)...")
        try:
            response = requests.get(self.MODEL_URL, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading ESRGAN model") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Model downloaded successfully to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download Real-ESRGAN model: {e}")
            return False
    
    def _load_model(self):
        """Load the Real-ESRGAN model with pretrained weights."""
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / self.MODEL_FILENAME
        
        # Download if not present
        if not model_path.exists():
            success = self._download_model(str(model_path))
            if not success:
                logger.warning("Could not download Real-ESRGAN model")
                return None
        
        try:
            # Build the RRDB network
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
            
            # Load pretrained weights
            loadnet = torch.load(str(model_path), map_location=self.device, weights_only=False)
            
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            elif 'params' in loadnet:
                keyname = 'params'
            else:
                keyname = None
            
            if keyname:
                model.load_state_dict(loadnet[keyname], strict=True)
            else:
                model.load_state_dict(loadnet, strict=True)
            
            model.eval()
            model = model.to(self.device)
            
            logger.info("✓ Real-ESRGAN model loaded successfully (RRDB x4plus)")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN weights: {e}")
            return None
    
    def _to_tensor(self, img):
        """Convert BGR numpy image to normalized tensor."""
        img = img.astype(np.float32) / 255.0
        # BGR to RGB, HWC to CHW
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        return img.unsqueeze(0).to(self.device)
    
    def _to_image(self, tensor):
        """Convert tensor back to BGR numpy image."""
        output = tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # CHW to HWC, RGB to BGR
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        return (output * 255.0).round().astype(np.uint8)
    
    @torch.no_grad()
    def _enhance_tiled(self, img):
        """Process image in tiles to manage memory on CPU."""
        img_tensor = self._to_tensor(img)
        _, _, height, width = img_tensor.shape
        
        out_h = height * self.scale
        out_w = width * self.scale
        
        # Small images: process directly
        if height <= self.tile_size and width <= self.tile_size:
            return self._to_image(self.model(img_tensor))
        
        # Tile processing for larger images
        output = torch.zeros(1, 3, out_h, out_w, device=self.device)
        
        tiles_y = max(1, (height + self.tile_size - 1) // self.tile_size)
        tiles_x = max(1, (width + self.tile_size - 1) // self.tile_size)
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Input tile with padding
                x0 = tx * self.tile_size
                y0 = ty * self.tile_size
                x1 = min(x0 + self.tile_size, width)
                y1 = min(y0 + self.tile_size, height)
                
                # Add padding
                px0 = max(x0 - self.tile_pad, 0)
                py0 = max(y0 - self.tile_pad, 0)
                px1 = min(x1 + self.tile_pad, width)
                py1 = min(y1 + self.tile_pad, height)
                
                tile_in = img_tensor[:, :, py0:py1, px0:px1]
                tile_out = self.model(tile_in)
                
                # Output coordinates (scaled)
                ox0 = (x0 - px0) * self.scale
                oy0 = (y0 - py0) * self.scale
                ox1 = ox0 + (x1 - x0) * self.scale
                oy1 = oy0 + (y1 - y0) * self.scale
                
                # Place in output
                out_ox0 = x0 * self.scale
                out_oy0 = y0 * self.scale
                out_ox1 = min(x1 * self.scale, out_w)
                out_oy1 = min(y1 * self.scale, out_h)
                
                copy_w = min(ox1 - ox0, out_ox1 - out_ox0, tile_out.shape[3] - ox0)
                copy_h = min(oy1 - oy0, out_oy1 - out_oy0, tile_out.shape[2] - oy0)
                
                output[:, :, out_oy0:out_oy0+copy_h, out_ox0:out_ox0+copy_w] = \
                    tile_out[:, :, oy0:oy0+copy_h, ox0:ox0+copy_w]
        
        return self._to_image(output)
    
    def enhance_image(self, image):
        """
        Enhance a single image using Real-ESRGAN.
        
        For CPU processing, we resize large inputs to a manageable size first,
        then apply the 4x ESRGAN model.
        
        Args:
            image: BGR numpy array
            
        Returns:
            Enhanced BGR numpy array
        """
        if self.model is not None:
            try:
                h, w = image.shape[:2]
                
                # On CPU, limit input size to avoid very long processing
                max_input_dim = (
                    config.GAN_MAX_INPUT_DIM_CPU if self.device.type == 'cpu' else 1024
                )
                if max(h, w) > max_input_dim:
                    ratio = max_input_dim / max(h, w)
                    new_h, new_w = int(h * ratio), int(w * ratio)
                    img_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    img_small = image
                
                result = self._enhance_tiled(img_small)
                
                if result is not None:
                    logger.debug(f"ESRGAN: {image.shape[:2]} → {img_small.shape[:2]} → {result.shape[:2]}")
                    return result
                
            except Exception as e:
                logger.warning(f"ESRGAN enhancement failed: {e}")
        
        # Fallback
        logger.info("Using fallback enhancement (CLAHE + Unsharp + Bilateral)")
        return self._fallback_enhance(image)
    
    def _fallback_enhance(self, image):
        """High-quality fallback enhancement."""
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w * self.scale, h * self.scale),
                             interpolation=cv2.INTER_LANCZOS4)
        
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
        enhanced = cv2.addWeighted(enhanced, 1.6, gaussian, -0.6, 0)
        enhanced = cv2.bilateralFilter(enhanced, 7, 60, 60)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def enhance_batch(self, image_paths, output_dir):
        """Enhance a batch of images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for image_path in tqdm(image_paths, desc="Enhancing with Real-ESRGAN"):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Cannot read image: {image_path}")
                    continue
                
                enhanced = self.enhance_image(image)
                
                image_path = Path(image_path)
                filename = (
                    f"{image_path.parent.name}_{image_path.stem.replace('_raw', '_enhanced')}.jpg"
                )
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                output_paths.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Error enhancing {image_path}: {str(e)}")
        
        return output_paths


def enhance_all_frames(extracted_frames_info, output_dir, upscale_factor=4):
    """Enhance all extracted frames using Real-ESRGAN."""
    enhancer = RealESRGANEnhancer(scale=upscale_factor)
    
    image_paths = [info['path'] for info in extracted_frames_info]
    enhanced_paths = enhancer.enhance_batch(image_paths, output_dir)
    
    logger.info(f"Enhanced {len(enhanced_paths)} frames using Real-ESRGAN")
    return enhanced_paths
