import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFile

# Allow loading images even if stream ended without full EOF block
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def _to_numpy(img: Any) -> Optional[np.ndarray]:
    """Convert input to numpy array if PIL Image, array, or filepath/dict."""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img)
    if isinstance(img, dict) and "path" in img:
        try:
            return np.array(Image.open(img["path"]))
        except Exception as e:
            logger.warning(f"Failed to open image from dict path {img.get('path')}: {e}")
            return None
    if isinstance(img, (str, os.PathLike)) and os.path.exists(img):
        try:
            return np.array(Image.open(img))
        except Exception as e:
            logger.warning(f"Failed to open image from path {img}: {e}")
            return None
    return None


def build_mask_from_editor(
    editor_data: Any,
    target_shape: Optional[Tuple[int, int]] = None,
    dilation_px: int = 4
) -> np.ndarray:
    """
    Extract a binary mask (H, W) of uint8 (0 or 255) from Gradio ImageEditor output.
    target_shape: Optional (height, width) to match original video frame dimensions.
    dilation_px: Pixels to dilate mask to ensure watermark edges are covered.
    """
    if editor_data is None:
        if target_shape:
            return np.zeros(target_shape, dtype=np.uint8)
        return np.zeros((100, 100), dtype=np.uint8)

    h, w = (target_shape if target_shape else (100, 100))
    raw_mask = None

    # Handle dictionary output from gr.ImageEditor
    if isinstance(editor_data, dict):
        layers = editor_data.get("layers", [])
        background = _to_numpy(editor_data.get("background"))
        composite = _to_numpy(editor_data.get("composite"))

        # 1. First priority: Check layers where user drew strokes
        if layers and len(layers) > 0:
            layer_masks = []
            for lyr in layers:
                arr = _to_numpy(lyr)
                if arr is None:
                    continue
                if arr.ndim == 3 and arr.shape[2] == 4:
                    # RGBA: where alpha > 10
                    alpha = arr[:, :, 3]
                    layer_masks.append(alpha > 10)
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    # RGB: any non-black pixel
                    layer_masks.append(np.any(arr > 10, axis=-1))
                elif arr.ndim == 2:
                    layer_masks.append(arr > 10)

            if layer_masks:
                combined_bool = np.logical_or.reduce(layer_masks)
                if np.any(combined_bool):
                    raw_mask = (combined_bool.astype(np.uint8)) * 255

        # 2. Second priority: If no layer mask, compare composite with background
        if raw_mask is None or not np.any(raw_mask > 0):
            if composite is not None and background is not None:
                # Ensure same shape
                if composite.shape == background.shape:
                    diff = np.abs(composite.astype(np.int16) - background.astype(np.int16))
                    # Check RGB channels for significant difference
                    diff_rgb = diff[:, :, :3] if diff.ndim == 3 else diff
                    diff_mask = np.any(diff_rgb > 20, axis=-1)
                    if np.any(diff_mask):
                        raw_mask = (diff_mask.astype(np.uint8)) * 255

        # 3. Third priority: Direct "mask" key if provided by Gradio
        if (raw_mask is None or not np.any(raw_mask > 0)) and "mask" in editor_data:
            m = _to_numpy(editor_data["mask"])
            if m is not None:
                if m.ndim == 3:
                    raw_mask = (m[:, :, 0] > 10).astype(np.uint8) * 255
                else:
                    raw_mask = (m > 10).astype(np.uint8) * 255

    # Handle direct numpy array or PIL Image
    elif isinstance(editor_data, (np.ndarray, Image.Image)):
        arr = _to_numpy(editor_data)
        if arr is not None:
            if arr.ndim == 3:
                raw_mask = (arr[:, :, 0] > 10).astype(np.uint8) * 255
            else:
                raw_mask = (arr > 10).astype(np.uint8) * 255

    if raw_mask is None:
        raw_mask = np.zeros((h, w), dtype=np.uint8)

    # Resize mask to target video resolution if needed
    if target_shape is not None:
        target_h, target_w = target_shape
        if raw_mask.shape != (target_h, target_w):
            raw_mask = cv2.resize(
                raw_mask,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

    # Dilate mask if requested
    if dilation_px > 0 and np.any(raw_mask > 0):
        ksize = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        raw_mask = cv2.dilate(raw_mask, kernel, iterations=1)

    return raw_mask


def extract_bounding_boxes(
    mask: np.ndarray,
    min_area: int = 16,
    max_boxes: int = 10
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes (x, y, w, h) for all connected components in mask.
    Returns:
        List of tuples: [(x, y, w, h), ...]
    """
    if mask is None or not np.any(mask > 0):
        return []

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Clamp to image boundaries
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))
            boxes.append((x, y, bw, bh))

    # If no contours met min_area, check global non-zero pixels
    if not boxes and np.any(mask > 0):
        ys, xs = np.where(mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        boxes.append((x_min, y_min, max(1, x_max - x_min + 1), max(1, y_max - y_min + 1)))

    # Sort boxes by area descending and limit
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_boxes]


def create_rect_mask(
    shape: Tuple[int, int],
    rects: List[Tuple[int, int, int, int]],
    dilation_px: int = 0
) -> np.ndarray:
    """Generate a binary mask (H, W) from a list of rectangles [(x, y, w, h), ...]."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in rects:
        x1 = max(0, min(x, w - 1))
        y1 = max(0, min(y, h - 1))
        x2 = max(x1 + 1, min(x1 + bw, w))
        y2 = max(y1 + 1, min(y1 + bh, h))
        mask[y1:y2, x1:x2] = 255

    if dilation_px > 0 and np.any(mask > 0):
        ksize = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def overlay_mask_on_frame(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45
) -> np.ndarray:
    """Overlay highlighted mask onto RGB frame for preview visualization."""
    if frame_rgb is None:
        return None
    overlay = frame_rgb.copy()
    if mask is None or not np.any(mask > 0):
        return overlay

    if mask.shape[:2] != frame_rgb.shape[:2]:
        mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(frame_rgb)
    colored_mask[:] = color

    mask_bool = mask > 0
    overlay[mask_bool] = (
        (1.0 - alpha) * frame_rgb[mask_bool] + alpha * colored_mask[mask_bool]
    ).astype(np.uint8)

    # Also draw green bounding boxes around detected rectangles
    boxes = extract_bounding_boxes(mask)
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 255, 50), 2)

    return overlay


def refine_mask_to_watermark_strokes(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    dilation_px: int = 4
) -> np.ndarray:
    """
    Refine a rough bounding box or brush mask into a tight mask covering only
    the actual watermark strokes/symbols. Preserves 60-80% of surrounding real
    background pixels (cobblestones, grass, textures) to eliminate blurry rectangles.
    """
    if frame_rgb is None or mask is None or not np.any(mask > 0):
        return mask

    h, w = frame_rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if not boxes:
        boxes = extract_bounding_boxes(mask)

    if not boxes:
        return mask

    refined_mask = np.zeros((h, w), dtype=np.uint8)

    for (bx, by, bw, bh) in boxes:
        # Padded ROI
        x1 = max(0, bx - 4)
        y1 = max(0, by - 4)
        x2 = min(w, bx + bw + 4)
        y2 = min(h, by + bh + 4)

        roi_rgb = frame_rgb[y1:y2, x1:x2]
        if roi_rgb.size == 0:
            continue
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Background approximation via median filter
        k_med = max(7, min(21, (min(roi_gray.shape) // 4) | 1))
        bg_blur = cv2.medianBlur(roi_gray, k_med)
        diff = cv2.absdiff(roi_gray, bg_blur)

        # 2. High brightness threshold (for white watermarks/logos)
        _, bright = cv2.threshold(roi_gray, 160, 255, cv2.THRESH_BINARY)
        # 3. High gradient difference threshold
        _, grad_diff = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        # 4. Text stroke edges
        edges = cv2.Canny(roi_gray, 35, 110)

        raw_strokes = cv2.bitwise_or(grad_diff, bright)
        raw_strokes = cv2.bitwise_or(raw_strokes, edges)

        # Dilate strokes to safely cover anti-aliased font borders
        ksize = max(3, 2 * dilation_px + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated_strokes = cv2.dilate(raw_strokes, kernel, iterations=1)

        # Check coverage
        stroke_ratio = np.count_nonzero(dilated_strokes) / float(dilated_strokes.size)
        if 0.04 <= stroke_ratio <= 0.85:
            refined_mask[y1:y2, x1:x2] = cv2.bitwise_or(refined_mask[y1:y2, x1:x2], dilated_strokes)
        else:
            # Fallback to full box if stroke isolation is ambiguous
            refined_mask[by:by+bh, bx:bx+bw] = 255

    return refined_mask

