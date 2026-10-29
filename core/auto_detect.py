"""
Automatic Watermark & Text Detection Module.
Detects static logos, social media handles, and watermarks across video frames
and computes optimal dilation and inpainting padding automatically.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.masks import create_rect_mask
from core.video_io import extract_frame_rgb, get_video_metadata

logger = logging.getLogger(__name__)


def auto_detect_watermarks(
    video_path: Path,
    num_samples: int = 5,
    mode: str = "balanced"
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], int, int]:
    """
    Automatically detect watermark regions in a video using temporal stability,
    high-frequency text edge localization, and clustering.
    
    Returns:
        (mask, bounding_boxes, auto_dilation_px, auto_padding_px)
    """
    video_path = Path(video_path).resolve()
    meta = get_video_metadata(video_path)
    h, w = meta["height"], meta["width"]
    total_frames = meta["total_frames"]

    # 1. Sample frames across the video to detect static regions (skipping dark fade-ins)
    num_samples = max(4, min(num_samples, total_frames))
    step = max(1, total_frames // num_samples)
    candidate_indices = [min(total_frames - 1, i * step) for i in range(num_samples)]

    frames = []
    for idx in candidate_indices:
        try:
            f = extract_frame_rgb(video_path, idx)
            # Filter out black fade-ins/intro screens if possible
            if f is not None and (f.mean() >= 16.0 or len(frames) == 0):
                frames.append(f)
        except Exception as e:
            logger.warning(f"Failed to extract frame {idx} for detection: {e}")

    if not frames:
        frames = [extract_frame_rgb(video_path, 0)]

    # Pick the sharpest, best-lit frame as reference (instead of potentially black frame 0)
    def frame_quality(fr):
        gr = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gr, cv2.CV_64F).var() if fr.mean() >= 16.0 else -1.0

    ref_frame = max(frames, key=frame_quality)

    # 2. Compute temporal variance (watermarks are static across frames)
    if len(frames) > 1:
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
        std_map = np.std(grays, axis=0)  # (H, W)
        static_mask = std_map < 16.0
    else:
        static_mask = np.ones((h, w), dtype=bool)

    # 3. High-contrast edge and gradient detection on the best reference frame
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    edge_mask = mag > 24.0

    # 4. Color variance / contrast against local neighborhood
    ref_hsv = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2HSV)
    sat = ref_hsv[:, :, 1]
    val = ref_hsv[:, :, 2]
    high_chroma = (sat > 75) | (val > 170) | (val < 25)

    # Combine signals: must be static AND have high frequency edges or distinctive chroma
    candidate_pixels = static_mask & (edge_mask | (high_chroma & (mag > 14.0)))
    candidate_img = candidate_pixels.astype(np.uint8) * 255

    # 5. Morphological clustering to group letters and words into text lines
    # Horizontal kernel connects adjacent characters in words
    k_w = max(15, int(w * 0.04))
    k_h = max(3, int(h * 0.012))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    connected = cv2.morphologyEx(candidate_img, cv2.MORPH_CLOSE, morph_kernel)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    total_area = float(w * h)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_area = bw * bh

        # Filter criteria: realistic watermark dimensions
        # Exclude thin border strips touching the extreme edges
        touches_left = (x <= 6 and bw < int(w * 0.08))
        touches_right = ((x + bw >= w - 6) and bw < int(w * 0.08))
        touches_top = (y <= 6 and bh < int(h * 0.03))
        touches_bottom = ((y + bh >= h - 6) and bh < int(h * 0.03))
        if touches_left or touches_right or touches_top or touches_bottom:
            continue

        # Real text / logos have decent dimensions
        if bw >= 24 and 10 <= bh <= int(h * 0.35) and box_area < (total_area * 0.15):
            # Exclude isolated tiny square specks in the middle that are texture noise
            aspect = bw / float(bh)
            if aspect < 1.0 and bw < 50 and bh < 50:
                continue

            # Expand box vertically and horizontally by safety margin to ensure logos / arcs are fully covered
            pad_y_top = max(4, int(bh * 0.18))
            pad_y_bot = max(4, int(bh * 0.12))
            pad_x = max(4, int(bw * 0.10))
            x_exp = max(0, x - pad_x)
            y_exp = max(0, y - pad_y_top)
            w_exp = min(w - x_exp, bw + 2 * pad_x)
            h_exp = min(h - y_exp, bh + pad_y_top + pad_y_bot)
            raw_boxes.append((x_exp, y_exp, w_exp, h_exp))

    # 6. Iterative merge of overlapping or horizontally adjacent boxes
    merged_boxes = _merge_nearby_boxes_iterative(
        raw_boxes,
        max_dist_x=int(w * 0.05),
        max_dist_y=int(h * 0.02)
    )

    # If no static boxes found with temporal filter, fallback to corner edge scan
    if not merged_boxes:
        merged_boxes = _fallback_corner_detection(ref_frame)


    # 7. Compute automatic Dilation and Padding based on detected box sizes
    if merged_boxes:
        avg_h = np.mean([b[3] for b in merged_boxes])
        avg_w = np.mean([b[2] for b in merged_boxes])
        auto_dilation = int(max(4, min(8, avg_h * 0.15)))
        auto_padding = int(max(20, min(45, max(avg_w, avg_h) * 0.35)))
    else:
        auto_dilation = 4
        auto_padding = 25

    mask = create_rect_mask((h, w), merged_boxes, dilation_px=auto_dilation)
    logger.info(f"Auto-detected {len(merged_boxes)} watermark regions: {merged_boxes}")

    return mask, merged_boxes, auto_dilation, auto_padding


def _boxes_intersect_or_adjacent(
    b1: Tuple[int, int, int, int],
    b2: Tuple[int, int, int, int],
    max_dist_x: int,
    max_dist_y: int
) -> bool:
    """Check if two boxes should be merged."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    # Horizontal proximity or overlap
    h_overlap = (max(x1, x2) <= min(x1 + w1, x2 + w2))
    h_dist = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    h_close = h_overlap or (h_dist <= max_dist_x)

    # Vertical proximity or overlap
    v_overlap_amount = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    min_h = min(h1, h2)
    v_close = (v_overlap_amount >= min_h * 0.3) or (abs(y1 - y2) <= max_dist_y) or (abs((y1 + h1) - (y2 + h2)) <= max_dist_y)

    return h_close and v_close


def _merge_pair(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Union two bounding boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    nx = min(x1, x2)
    ny = min(y1, y2)
    nx2 = max(x1 + w1, x2 + w2)
    ny2 = max(y1 + h1, y2 + h2)
    return (nx, ny, nx2 - nx, ny2 - ny)


def _merge_nearby_boxes_iterative(
    boxes: List[Tuple[int, int, int, int]],
    max_dist_x: int = 30,
    max_dist_y: int = 15
) -> List[Tuple[int, int, int, int]]:
    """Iteratively merge bounding boxes until convergence."""
    if not boxes:
        return []

    cur_boxes = list(boxes)
    merged = True

    while merged:
        merged = False
        new_list = []
        skip_indices = set()

        for i in range(len(cur_boxes)):
            if i in skip_indices:
                continue
            box_a = cur_boxes[i]
            for j in range(i + 1, len(cur_boxes)):
                if j in skip_indices:
                    continue
                box_b = cur_boxes[j]
                if _boxes_intersect_or_adjacent(box_a, box_b, max_dist_x, max_dist_y):
                    box_a = _merge_pair(box_a, box_b)
                    skip_indices.add(j)
                    merged = True
            new_list.append(box_a)

        cur_boxes = new_list

    # Sort top to bottom, left to right
    return sorted(cur_boxes, key=lambda b: (b[1], b[0]))


def _fallback_corner_detection(frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Inspect top-left, top-right, bottom-left, bottom-right for high-contrast watermark text."""
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    boxes = []

    # Check 4 corners (each 40% of width, 25% of height)
    corners = [
        ("top_left", 0, 0, int(w * 0.4), int(h * 0.25)),
        ("top_right", int(w * 0.6), 0, w - int(w * 0.6), int(h * 0.25)),
        ("bottom_left", 0, int(h * 0.75), int(w * 0.4), h - int(h * 0.75)),
        ("bottom_right", int(w * 0.6), int(h * 0.75), w - int(w * 0.6), h - int(h * 0.75)),
    ]

    for name, cx, cy, cw, ch in corners:
        roi = gray[cy:cy+ch, cx:cx+cw]
        grad = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5)))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw >= 40 and 12 <= bh <= (ch * 0.7):
                boxes.append((cx + x, cy + y, bw, bh))

    return _merge_nearby_boxes_iterative(boxes, max_dist_x=int(w * 0.05), max_dist_y=int(h * 0.02))

