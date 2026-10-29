"""
Pipeline orchestration module for Video Watermark Remover.
Manages multiprocessing for Standard Mode, sequential streaming for LaMa Mode,
Quick Mode integration, progress tracking, preview generation, and safe cleanup.
"""
import logging
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.lama_mode import LamaInpainter
from core.masks import extract_bounding_boxes
from core.quick_mode import run_quick_mode
from core.smart_mode import run_smart_ai_video
from core.standard_mode import inpaint_frame_standard
from core.video_io import (
    cleanup_job_dir,
    create_job_dir,
    extract_frame_rgb,
    get_video_metadata,
    merge_audio_and_finalize,
)

logger = logging.getLogger(__name__)

# Worker globals for multiprocessing (spawn-safe top-level)
_worker_mask = None
_worker_boxes = None
_worker_method = "telea"
_worker_padding = 20


def _init_standard_worker(mask: np.ndarray, boxes: List[Tuple[int, int, int, int]], method: str, padding: int) -> None:
    """Initializer for multiprocessing pool workers."""
    global _worker_mask, _worker_boxes, _worker_method, _worker_padding
    _worker_mask = mask
    _worker_boxes = boxes
    _worker_method = method
    _worker_padding = padding


def _standard_worker_process(frame_bgr: np.ndarray) -> np.ndarray:
    """Worker task to inpaint a single frame."""
    global _worker_mask, _worker_boxes, _worker_method, _worker_padding
    return inpaint_frame_standard(
        frame_bgr,
        _worker_mask,
        _worker_boxes,
        method_name=_worker_method,
        padding=_worker_padding
    )


def preview_single_frame(
    video_path: Path,
    frame_idx: int,
    mask: np.ndarray,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    mode: str = "Standard",
    standard_method: str = "telea",
    padding: int = 20,
    refine_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Process only ONE frame and return (before_rgb, after_rgb, info_text).
    """
    video_path = Path(video_path).resolve()
    orig_frame_rgb = extract_frame_rgb(video_path, frame_idx)
    h, w = orig_frame_rgb.shape[:2]

    if mask is None or not np.any(mask > 0):
        return orig_frame_rgb, orig_frame_rgb, "No watermark region selected."

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if not boxes:
        boxes = extract_bounding_boxes(mask)

    if refine_mask:
        from core.masks import refine_mask_to_watermark_strokes
        mask = refine_mask_to_watermark_strokes(orig_frame_rgb, mask, boxes, dilation_px=4)

    mode_lower = mode.lower()
    start_time = time.time()
    info = ""

    if "quick" in mode_lower:
        # For single frame quick preview, perform fast standard inpainting
        frame_bgr = cv2.cvtColor(orig_frame_rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = inpaint_frame_standard(
            frame_bgr, mask, boxes, method_name="telea", padding=padding
        )
        processed_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        elapsed = time.time() - start_time
        info = f"Quick Mode Preview (Telea estimate): {elapsed:.2f}s"

    elif "smart" in mode_lower or "high" in mode_lower or "lama" in mode_lower:
        from core.lama_mode import get_lama_inpainter
        inpainter = get_lama_inpainter()
        processed_rgb = inpainter.inpaint_frame(orig_frame_rgb, mask, boxes, padding=max(padding, 28))
        elapsed = time.time() - start_time
        label = "Smart AI (Neural)" if "smart" in mode_lower else "High Quality (LaMa)"
        info = f"{label} Preview ({inpainter.active_provider}): {elapsed:.2f}s"

    else:
        # Standard mode
        frame_bgr = cv2.cvtColor(orig_frame_rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = inpaint_frame_standard(
            frame_bgr, mask, boxes, method_name=standard_method, padding=padding
        )
        processed_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        elapsed = time.time() - start_time
        info = f"Standard Mode ({standard_method.upper()}): {elapsed:.2f}s"

    return orig_frame_rgb, processed_rgb, info


def process_video(
    input_video: Path,
    mask: np.ndarray,
    rectangles: Optional[List[Tuple[int, int, int, int]]] = None,
    mode: str = "Standard",
    standard_method: str = "telea",
    padding: int = 20,
    reference_frame_idx: Optional[int] = None,
    refine_mask: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Path:
    """
    Main orchestration pipeline for video watermark removal.
    """
    input_video = Path(input_video).resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # 1. Setup Job Directory and Output Directory
    job_dir = create_job_dir()
    logger.info(f"Created job directory: {job_dir}")

    output_dir = Path("./outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_stem = input_video.stem
    while clean_stem.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        clean_stem = Path(clean_stem).stem

    temp_video_path = job_dir / "temp_processed.mp4"
    final_output_path = output_dir / f"cleaned_{clean_stem}.mp4"

    try:
        # 2. Extract Metadata
        meta = get_video_metadata(input_video)
        width, height = meta["width"], meta["height"]
        fps = meta["fps"]
        total_frames = meta["total_frames"]
        has_audio = meta["has_audio"]

        # Ensure mask is sized to video resolution
        if mask is not None and mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        if not rectangles:
            rectangles = extract_bounding_boxes(mask) if mask is not None else []

        if not rectangles and (mask is None or not np.any(mask > 0)):
            raise ValueError("No watermark area or rectangles detected. Please draw over the watermark.")

        if refine_mask:
            from core.masks import refine_mask_to_watermark_strokes
            ref_idx = reference_frame_idx if reference_frame_idx is not None else 0
            try:
                ref_f = extract_frame_rgb(input_video, ref_idx)
                mask = refine_mask_to_watermark_strokes(ref_f, mask, rectangles, dilation_px=4)
            except Exception as e_ref:
                logger.warning(f"Could not refine watermark mask: {e_ref}")

        mode_clean = mode.lower()

        # 3. Smart AI Mode (Zero Blur + Flawless Neural Background + 60+ FPS)
        if "smart" in mode_clean:
            return run_smart_ai_video(
                input_video=input_video,
                output_video=final_output_path,
                mask=mask,
                rectangles=rectangles,
                reference_frame_idx=reference_frame_idx,
                padding=padding,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )


        # 4. Quick Mode execution
        elif "quick" in mode_clean:
            if progress_callback:
                progress_callback(0.05, "Starting Quick Mode (FFmpeg delogo)...")

            class CancelAdapter:
                def is_set(self):
                    return cancel_check() if cancel_check else False

            run_quick_mode(
                input_video=input_video,
                rectangles=rectangles,
                output_video=final_output_path,
                has_audio=has_audio,
                total_frames=total_frames,
                progress_callback=progress_callback,
                cancel_flag=CancelAdapter() if cancel_check else None
            )

            if progress_callback:
                progress_callback(1.0, "Completed!")
            return final_output_path

        # 4. Standard Mode (OpenCV Multiprocessing)
        elif "standard" in mode_clean:
            num_workers = max(1, (os.cpu_count() or 4) - 1)
            batch_size = max(4, num_workers * 2)

            if progress_callback:
                progress_callback(
                    0.02,
                    f"Initializing Standard Mode with {num_workers} parallel CPU workers..."
                )

            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {input_video} with OpenCV.")

            writer = cv2.VideoWriter(
                str(temp_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            pool = mp.Pool(
                processes=num_workers,
                initializer=_init_standard_worker,
                initargs=(mask, rectangles, standard_method, padding)
            )

            frames_processed = 0
            start_time = time.time()

            try:
                while True:
                    if cancel_check and cancel_check():
                        raise RuntimeError("Processing was cancelled by user.")

                    batch = []
                    for _ in range(batch_size):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        batch.append(frame)

                    if not batch:
                        break

                    results = pool.map(_standard_worker_process, batch)

                    for res_frame in results:
                        writer.write(res_frame)

                    frames_processed += len(batch)
                    elapsed = time.time() - start_time
                    fps_proc = frames_processed / elapsed if elapsed > 0 else 0
                    rem_frames = max(0, total_frames - frames_processed)
                    eta_sec = int(rem_frames / fps_proc) if fps_proc > 0 else 0

                    pct = min(0.95, frames_processed / float(total_frames))
                    if progress_callback:
                        progress_callback(
                            pct,
                            f"Standard Mode: {frames_processed}/{total_frames} frames ({int(pct*100)}%) | {fps_proc:.1f} fps | ETA: {eta_sec}s"
                        )

            finally:
                cap.release()
                writer.release()
                pool.close()
                pool.join()

        # 5. High Quality Mode (LaMa ONNX)
        else:
            if progress_callback:
                progress_callback(0.02, "Loading LaMa ONNX inpainter...")

            from core.lama_mode import get_lama_inpainter
            inpainter = get_lama_inpainter(progress_callback=progress_callback)
            provider_info = inpainter.active_provider

            if progress_callback:
                progress_callback(
                    0.05,
                    f"High Quality Mode running on {provider_info}..."
                )

            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {input_video} with OpenCV.")

            writer = cv2.VideoWriter(
                str(temp_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            frames_processed = 0
            start_time = time.time()

            try:
                while True:
                    if cancel_check and cancel_check():
                        raise RuntimeError("Processing was cancelled by user.")

                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    inpainted_rgb = inpainter.inpaint_frame(
                        frame_rgb,
                        mask,
                        rectangles,
                        padding=max(padding, 32)
                    )
                    out_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
                    writer.write(out_bgr)

                    frames_processed += 1
                    elapsed = time.time() - start_time
                    fps_proc = frames_processed / elapsed if elapsed > 0 else 0
                    rem_frames = max(0, total_frames - frames_processed)
                    eta_sec = int(rem_frames / fps_proc) if fps_proc > 0 else 0

                    pct = min(0.95, frames_processed / float(total_frames))
                    if progress_callback and frames_processed % 2 == 0:
                        progress_callback(
                            pct,
                            f"LaMa ({provider_info}): {frames_processed}/{total_frames} frames ({int(pct*100)}%) | {fps_proc:.1f} fps | ETA: {eta_sec}s"
                        )
            finally:
                cap.release()
                writer.release()

        # 6. Audio Merge and Finalize with FFmpeg
        if progress_callback:
            progress_callback(0.96, "Encoding final MP4 with H.264 & copying original audio...")

        merge_audio_and_finalize(
            processed_video_path=temp_video_path,
            original_video_path=input_video,
            output_path=final_output_path,
            has_audio=has_audio
        )

        # Clean up temporary job directory
        cleanup_job_dir(job_dir)

        if progress_callback:
            progress_callback(1.0, "Watermark removal complete!")

        return final_output_path

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Clean temp video on failure
        if temp_video_path.exists():
            temp_video_path.unlink(missing_ok=True)
        raise e
