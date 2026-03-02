"""
src/preprocessing/resolution_enforcement.py
Enforces minimum/maximum resolution constraints from preprocessing.yaml.
Upscales images below minimum and downscales images above maximum.
Optionally pads to a target aspect ratio.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from loguru import logger

from src.utils.config_loader import preprocessing_cfg as cfg
_re = cfg.resolution_enforcement

MIN_W: int = _re.min_width
MIN_H: int = _re.min_height
MAX_W: int = _re.max_width
MAX_H: int = _re.max_height
REJECT_BELOW_MIN: bool = _re.reject_below_min
PAD_TO_ASPECT: bool = _re.pad_to_aspect_ratio
PAD_COLOR: int = _re.pad_color
TARGET_W: int = cfg.image.target_width
TARGET_H: int = cfg.image.target_height

def enforce_resolution(image: np.ndarray) -> np.ndarray:

    if not _re.enabled:
        logger.debug("Resolution enforcement disabled in config, skipping.")
        return image

    h, w = image.shape[:2]

    if w < MIN_W or h < MIN_H:
        if REJECT_BELOW_MIN:
            raise ValueError(
                f"Image ({w}x{h}) is below minimum resolution "
                f"({MIN_W}x{MIN_H}). Set reject_below_min=false to upscale."
            )
        image = _resize_to_fit(image, min_w=MIN_W, min_h=MIN_H, mode="upscale")
        h, w = image.shape[:2]
        logger.debug(f"Upscaled to minimum resolution: ({w}x{h})")

    if w > MAX_W or h > MAX_H:
        image = _resize_to_fit(image, max_w=MAX_W, max_h=MAX_H, mode="downscale")
        h, w = image.shape[:2]
        logger.debug(f"Downscaled to maximum resolution: ({w}x{h})")

    if PAD_TO_ASPECT:
        image = _pad_to_aspect(image, TARGET_W, TARGET_H, fill=PAD_COLOR)
        h, w = image.shape[:2]
        logger.debug(f"Padded to aspect ratio: ({w}x{h})")

    return image

def _resize_to_fit(
    image: np.ndarray,
    min_w: int = 0,
    min_h: int = 0,
    max_w: int = 99999,
    max_h: int = 99999,
    mode: str = "downscale",
) -> np.ndarray:
    """Resize while preserving aspect ratio to satisfy constraints."""
    h, w = image.shape[:2]
    if mode == "upscale":
        scale = max(min_w / w, min_h / h)
    else:
        scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil_img = _to_pil(image)
    return np.array(pil_img.resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)


def _pad_to_aspect(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    fill: int = 255,
) -> np.ndarray:
    h, w = image.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_h = int(round(w / target_ratio))
        pad_top = (new_h - h) // 2
        pad_bot = new_h - h - pad_top
        pad = ((pad_top, pad_bot), (0, 0)) if image.ndim == 2 else ((pad_top, pad_bot), (0, 0), (0, 0))
    else:
        new_w = int(round(h * target_ratio))
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        pad = ((0, 0), (pad_left, pad_right)) if image.ndim == 2 else ((0, 0), (pad_left, pad_right), (0, 0))

    return np.pad(image, pad, mode="constant", constant_values=fill)


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image, mode="L" if image.ndim == 2 else "RGB")
