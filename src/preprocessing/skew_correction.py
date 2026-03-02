"""
src/preprocessing/skew_correction.py
Detects and corrects document skew using the Hough line transform
or projection profile method. Config from preprocessing.yaml.
"""

from __future__ import annotations

import numpy as np
import cv2
from loguru import logger

from src.utils.config_loader import preprocessing_cfg as cfg

_sc = cfg.skew_correction

METHOD: str = _sc.method
MAX_SKEW: float = _sc.max_skew_angle_deg
BG_FILL: int = _sc.background_fill
HOUGH_THRESHOLD: int = _sc.hough.threshold
HOUGH_MIN_LINE: int = _sc.hough.min_line_length
HOUGH_MAX_GAP: int = _sc.hough.max_line_gap



def correct_skew(image: np.ndarray) -> np.ndarray:
    if not _sc.enabled:
        logger.debug("Skew correction disabled in config, skipping.")
        return image

    if METHOD == "hough":
        angle = _detect_skew_hough(image)
    elif METHOD == "projection_profile":
        angle = _detect_skew_projection(image)
    else:
        raise ValueError(f"Unknown skew method: {METHOD}")

    if abs(angle) < 0.1:
        logger.debug("No significant skew detected.")
        return image

    if abs(angle) > MAX_SKEW:
        logger.warning(
            f"Detected skew angle {angle:.2f}° exceeds max ({MAX_SKEW}°). "
            "Image may have severe misalignment."
        )

    corrected = _rotate_image(image, angle)
    logger.debug(f"Skew corrected by {angle:.2f}°")
    return corrected

def _detect_skew_hough(image: np.ndarray) -> float:
    """Estimate skew angle using Hough line transform."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is None:
        logger.debug("Hough: no lines detected, assuming 0° skew.")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines
        if abs(angle) < MAX_SKEW:
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    return median_angle


def _detect_skew_projection(image: np.ndarray) -> float:
    best_angle = 0.0
    best_variance = -1.0
    binarised = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    for angle in np.linspace(-MAX_SKEW, MAX_SKEW, 180):
        rotated = _rotate_image(binarised, angle)
        projection = np.sum(rotated, axis=1).astype(np.float32)
        variance = float(np.var(projection))
        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    return best_angle

def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its centre, filling borders with BG_FILL."""
    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BG_FILL,
    )
    return rotated
