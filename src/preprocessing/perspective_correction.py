"""
src/preprocessing/perspective_correction.py
Corrects perspective distortion (e.g. photo taken at an angle) by
detecting the cheque boundary contour and applying a four-point warp.
Config from preprocessing.yaml.
"""

from __future__ import annotations

import numpy as np
import cv2
from loguru import logger

from src.utils.config_loader import preprocessing_cfg as cfg

_pc = cfg.perspective_correction

MIN_AREA_RATIO: float = _pc.min_contour_area_ratio
APPROX_EPSILON: float = _pc.approx_epsilon_factor
TARGET_W: int = cfg.image.target_width
TARGET_H: int = cfg.image.target_height

def correct_perspective(image: np.ndarray) -> np.ndarray:
    if not _pc.enabled:
        logger.debug("Perspective correction disabled in config, skipping.")
        return image

    quad = _find_document_quad(image)
    if quad is None:
        logger.warning("Perspective correction: no valid quad found. Returning original.")
        return image

    corrected = _four_point_transform(image, quad)
    logger.debug(f"Perspective corrected → output size ({corrected.shape[1]}x{corrected.shape[0]})")
    return corrected

def _find_document_quad(image: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    total_area = h * w

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < total_area * MIN_AREA_RATIO:
            break
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPSILON * perimeter, closed=True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H))
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect
