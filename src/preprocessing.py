"""
preprocessing.py
----------------
Image preprocessing utilities for the Flamingo/IDEFICS2 medical pipeline.

Handles:
  - Loading images from file paths or URLs
  - Grayscale-to-RGB conversion (X-rays, CT)
  - CLAHE contrast enhancement for better radiograph visibility
  - Resizing + padding to target resolution
  - Tensor normalization compatible with CLIP/SigLIP visual encoders
  - Batch preprocessing
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from loguru import logger
from PIL import Image, ImageOps

from .config import PreprocessingConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Stateless image preprocessor.  All methods are pure functions wrapped
    in a class for testability and dependency injection.

    Parameters
    ----------
    config : PreprocessingConfig
    """

    def __init__(self, config: PreprocessingConfig):
        self.cfg = config

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def load_and_preprocess(
        self,
        source: Union[str, Path, bytes, Image.Image],
        enhance_contrast: Optional[bool] = None,
    ) -> Image.Image:
        """
        Full pipeline: load → convert → enhance → resize.

        Parameters
        ----------
        source :
            File path (str/Path), URL string (http/https), raw bytes, or
            an already-opened PIL Image.
        enhance_contrast :
            Override config.apply_clahe.  Pass False for non-radiograph images.

        Returns
        -------
        PIL.Image  in RGB mode, resized to config.target_size.
        """
        pil_img = self._load(source)
        pil_img = self._to_rgb(pil_img)

        apply_clahe = enhance_contrast if enhance_contrast is not None else self.cfg.apply_clahe
        if apply_clahe:
            pil_img = self._apply_clahe(pil_img)

        pil_img = self._resize_with_pad(pil_img, self.cfg.target_size)
        return pil_img

    def load_and_preprocess_batch(
        self,
        sources: List[Union[str, Path, bytes, Image.Image]],
        enhance_contrast: Optional[bool] = None,
    ) -> List[Image.Image]:
        """Batch version — applies load_and_preprocess to every item."""
        results = []
        for idx, src in enumerate(sources):
            try:
                results.append(self.load_and_preprocess(src, enhance_contrast))
            except Exception as exc:
                logger.warning(f"Failed to preprocess image {idx}: {exc}")
                results.append(self._placeholder_image())
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Load a PIL Image from various source types."""
        if isinstance(source, Image.Image):
            return source.copy()
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source))
        source_str = str(source)
        if source_str.startswith("http://") or source_str.startswith("https://"):
            response = requests.get(source_str, timeout=15)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path)

    def _to_rgb(self, img: Image.Image) -> Image.Image:
        """Convert any mode to RGB (handles grayscale X-rays, RGBA PNGs, etc.)."""
        if img.mode == "RGB":
            return img
        if img.mode in ("L", "LA") and self.cfg.convert_grayscale_to_rgb:
            # Grayscale → RGB by repeating the channel
            return ImageOps.grayscale(img).convert("RGB")
        if img.mode == "RGBA":
            # Composite onto white background
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert("RGB")

    def _apply_clahe(self, img: Image.Image) -> Image.Image:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV.
        Converts to LAB color space, applies CLAHE to L-channel only (preserves hue).

        Particularly useful for:
        - Chest X-rays (enhances pulmonary vascular markings)
        - CT scans
        - Low-contrast medical images
        """
        cv_img = np.array(img)                          # H×W×3, uint8
        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_grid,
        )
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_enhanced)

    @staticmethod
    def _resize_with_pad(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to target_size maintaining aspect ratio,
        then pad with black pixels to fill exactly target_size.
        This avoids distortion — important for medical images.
        """
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # Compute scale maintaining aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad to target size
        pad_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        pad_img.paste(img_resized, (offset_x, offset_y))
        return pad_img

    @staticmethod
    def _placeholder_image() -> Image.Image:
        """Return a black placeholder image when loading fails."""
        warnings.warn("Using placeholder (black) image due to load failure.")
        return Image.new("RGB", (448, 448), (0, 0, 0))
