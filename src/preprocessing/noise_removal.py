"""
src/preprocessing/noise_removal.py
Applies configurable noise removal filters to cheque images.
All filter parameters come from preprocessing.yaml → noise_removal.
"""

from __future__ import annotations

import numpy as np
import cv2
from loguru import logger
from src.utils.config_loader import preprocessing_cfg as cfg

_nr = cfg.noise_removal

GAUSSIAN_ENABLED: bool = _nr.gaussian_blur.enabled
GAUSSIAN_K: int = _nr.gaussian_blur.kernel_size
GAUSSIAN_SIGMA: int = _nr.gaussian_blur.sigma

MEDIAN_ENABLED: bool = _nr.median_blur.enabled
MEDIAN_K: int = _nr.median_blur.kernel_size

BILATERAL_ENABLED: bool = _nr.bilateral_filter.enabled
BILATERAL_D: int = _nr.bilateral_filter.d
BILATERAL_SC: int = _nr.bilateral_filter.sigma_color
BILATERAL_SS: int = _nr.bilateral_filter.sigma_space

MORPH_ENABLED: bool = _nr.morphological.enabled
MORPH_OP: str = _nr.morphological.operation
MORPH_K: int = _nr.morphological.kernel_size
MORPH_ITER: int = _nr.morphological.iterations


def remove_noise(image: np.ndarray) -> np.ndarray:
    if not _nr.enabled:
        logger.debug("Noise removal disabled in config, skipping.")
        return image
    
    result = image.copy()

    if GAUSSIAN_ENABLED:
        k = _ensure_odd(GAUSSIAN_K)
        result = cv2.GaussianBlur(result, (k, k), GAUSSIAN_SIGMA)
        logger.debug(f"Gaussian blur applied (k={k}, σ={GAUSSIAN_SIGMA})")

    if MEDIAN_ENABLED:
        k = _ensure_odd(MEDIAN_K)
        result = cv2.medianBlur(result, k)
        logger.debug(f"Median blur applied (k={k})")

    if BILATERAL_ENABLED:
        result = cv2.bilateralFilter(result, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)
        logger.debug(f"Bilateral filter applied (d={BILATERAL_D})")

    if MORPH_ENABLED:
        result = _apply_morphological(result)

    return result

def _apply_morphological(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_K, MORPH_K)
    )
    if MORPH_OP == "opening":
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)
    elif MORPH_OP == "closing":
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
    elif MORPH_OP == "both":
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
    else:
        raise ValueError(f"Unknown morphological operation: {MORPH_OP}")
    logger.debug(f"Morphological {MORPH_OP} applied (k={MORPH_K}, iter={MORPH_ITER})")
    return result


def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1
