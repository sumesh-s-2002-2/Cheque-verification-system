"""
src/preprocessing/dpi_normalization.py
Resamples a cheque image from its native DPI to the target DPI
defined in preprocessing.yaml → image.target_dpi.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from loguru import logger

from src.utils.config_loader import preprocessing_cfg as cfg

TARGET_DPI: int = cfg.image.target_dpi
RESAMPLE_FILTER_NAME: str = cfg.dpi_normalization.resample_filter
DEFAULT_DPI: int = cfg.dpi_normalization.default_assumed_dpi

_RESAMPLE_MAP = {
    "LANCZOS": Image.LANCZOS,
    "BICUBIC": Image.BICUBIC,
    "BILINEAR": Image.BILINEAR,
    "NEAREST": Image.NEAREST,
}

RESAMPLE_FILTER = _RESAMPLE_MAP.get(RESAMPLE_FILTER_NAME, Image.LANCZOS)

def normalize_dpi(
    image: np.ndarray,
    source_dpi: int,
    target_dpi: int = TARGET_DPI,
) -> np.ndarray:
    
    if not cfg.dpi_normalization.enabled:
        logger.debug("DPI normalization disabled in config, skipping.")
        return image

    if source_dpi == target_dpi:
        logger.debug(f"Source DPI ({source_dpi}) == target DPI, no resampling needed.")
        return image

    scale = target_dpi / source_dpi
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    pil_img = _to_pil(image)
    resampled = pil_img.resize((new_w, new_h), resample=RESAMPLE_FILTER)
    result = np.array(resampled, dtype=np.uint8)

    logger.debug(
        f"DPI normalised: {source_dpi} → {target_dpi} dpi | "
        f"({w}x{h}) → ({new_w}x{new_h})"
    )
    return result

def _to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    return Image.fromarray(image, mode="RGB")
