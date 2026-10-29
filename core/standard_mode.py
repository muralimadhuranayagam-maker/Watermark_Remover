"""
Standard Mode: OpenCV inpainting (Telea / Navier-Stokes) with cropped region optimization.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.masks import extract_bounding_boxes

logger = logging.getLogger(__name__)


def inpaint_frame_standard(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    method_name: str = "telea",
    padding: int = 20,
    radius: int = 3
) -> np.ndarray:
    """
    Inpaint watermark regions on a single frame using OpenCV inpainting.
    Applies inpainting only to a cropped region around each mask plus padding (~20px),
    then pastes the result back into the frame.
    """
    if mask is None or not np.any(mask > 0):
        return frame_bgr

    output_frame = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    flags = cv2.INPAINT_NS if str(method_name).lower() == "ns" else cv2.INPAINT_TELEA

    if boxes is None or len(boxes) == 0:
        boxes = extract_bounding_boxes(mask)

    if not boxes:
        # Full frame inpainting fallback if no distinct boxes found
        return cv2.inpaint(output_frame, mask, inpaintRadius=radius, flags=flags)

    for (bx, by, bw, bh) in boxes:
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(w, bx + bw + padding)
        y2 = min(h, by + bh + padding)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = output_frame[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        if not np.any(crop_mask > 0):
            continue

        inpainted_crop = cv2.inpaint(crop_img, crop_mask, inpaintRadius=radius, flags=flags)
        output_frame[y1:y2, x1:x2] = inpainted_crop

    return output_frame
