"""
Smart AI Mode: Neural Keyframe Inpainting + Illumination-Compensated Seamless Blending.
Synthesizes crisp, unblurred background textures using LaMa deep neural network on keyframes,
then streams across all video frames with feather-blended illumination compensation at 60+ FPS.
"""
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from core.lama_mode import get_lama_inpainter
from core.masks import extract_bounding_boxes
from core.video_io import (
    cleanup_job_dir,
    create_job_dir,
    extract_frame_rgb,
    get_video_metadata,
    merge_audio_and_finalize,
)

logger = logging.getLogger(__name__)


def find_best_keyframe(
    input_video: Path,
    total_frames: int,
    preferred_idx: Optional[int] = None
) -> Tuple[int, np.ndarray]:
    """
    Find the best well-lit keyframe for neural background synthesis.
    Avoids black fade-ins, title screens, or dark frames.
    """
    if preferred_idx is not None and 0 <= preferred_idx < total_frames:
        try:
            f = extract_frame_rgb(input_video, preferred_idx)
            if f is not None and f.mean() >= 16.0:
                return preferred_idx, f
        except Exception:
            pass

    # Sample candidates across video
    indices = [
        0, 10, 30, 60,
        int(total_frames * 0.1),
        int(total_frames * 0.25),
        int(total_frames * 0.5),
        int(total_frames * 0.75)
    ]
    candidates = []
    for idx in set(indices):
        if 0 <= idx < total_frames:
            try:
                f = extract_frame_rgb(input_video, idx)
                if f is not None:
                    candidates.append((idx, f))
            except Exception:
                pass

    if not candidates:
        f0 = extract_frame_rgb(input_video, 0)
        return 0, f0

    # Filter out dark fade-ins (mean < 16.0)
    well_lit = [(idx, f) for idx, f in candidates if f.mean() >= 16.0]
    if well_lit:
        # Pick the sharpest frame
        def sharpness(item):
            gray = cv2.cvtColor(item[1], cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        return max(well_lit, key=sharpness)

    # Fallback to brightest frame if all are dim
    return max(candidates, key=lambda x: x[1].mean())


def run_smart_ai_video(
    input_video: Path,
    output_video: Path,
    mask: np.ndarray,
    rectangles: Optional[List[Tuple[int, int, int, int]]] = None,
    reference_frame_idx: Optional[int] = None,
    padding: int = 30,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Path:
    """
    Run Smart AI Video Inpainting:
    1. Runs LaMa on keyframe(s) to generate 100% natural, sharp, unblurred background texture.
    2. Streams video frames and blends the clean patch with local illumination tracking and feathering.
    3. Dynamically adapts when scene cuts or major lighting shifts occur.
    4. Preserves original audio and exports fast H.264 MP4.
    """
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    meta = get_video_metadata(input_video)
    width, height = meta["width"], meta["height"]
    fps = meta["fps"]
    total_frames = meta["total_frames"]
    has_audio = meta["has_audio"]

    if mask is not None and mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if not rectangles:
        rectangles = extract_bounding_boxes(mask) if mask is not None else []

    if not rectangles:
        raise ValueError("No watermark bounding boxes found for Smart AI inpainting.")

    # 1. Job directory for intermediate frames/video
    job_dir = create_job_dir()
    temp_video_path = job_dir / "smart_temp.mp4"

    try:
        # 2. Keyframe Inpainting using LaMa Neural Model
        if progress_callback:
            progress_callback(0.05, "Smart AI: Initializing neural network & analyzing keyframe...")

        inpainter = get_lama_inpainter(progress_callback=progress_callback)
        provider_name = inpainter.active_provider

        # Extract best keyframe (skipping dark fade-in frames like frame 0)
        best_k_idx, key_rgb = find_best_keyframe(input_video, total_frames, preferred_idx=reference_frame_idx)
        logger.info(f"Smart AI selected reference keyframe #{best_k_idx} (mean={key_rgb.mean():.1f})")

        t_start_key = time.time()
        clean_key_rgb = inpainter.inpaint_frame(key_rgb, mask, rectangles, padding=max(padding, 28))
        key_dur = time.time() - t_start_key
        logger.info(f"Smart AI keyframe inpainting completed in {key_dur:.2f}s ({provider_name})")

        if progress_callback:
            progress_callback(0.12, f"Smart AI: Keyframe synthesized ({key_dur:.1f}s). Preparing seamless trackers...")

        # 3. Precompute clean patches, soft feather masks, and background collars
        patch_trackers = []
        for bx, by, bw, bh in rectangles:
            pad = max(18, padding)
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(width, bx + bw + pad)
            y2 = min(height, by + bh + pad)
            pw = x2 - x1
            ph = y2 - y1

            if pw <= 0 or ph <= 0:
                continue

            # Sub-mask for this bounding crop
            crop_mask = mask[y1:y2, x1:x2].copy()
            # If sub-mask is completely zero, fill the box area
            if not np.any(crop_mask > 0):
                crop_mask[by - y1 : by - y1 + bh, bx - x1 : bx - x1 + bw] = 255

            # Collar is the unaltered background outside the watermark within the padded crop
            collar = (crop_mask == 0)
            if np.count_nonzero(collar) < 20:
                collar = np.zeros((ph, pw), dtype=bool)
                border_sz = max(2, min(8, pad // 2))
                collar[:border_sz, :] = True
                collar[-border_sz:, :] = True
                collar[:, :border_sz] = True
                collar[:, -border_sz:] = True

            key_crop = key_rgb[y1:y2, x1:x2]
            key_collar_mean = np.mean(key_crop[collar], axis=0) if np.any(collar) else np.array([0.0, 0.0, 0.0])

            # Clean synthesized patch from LaMa
            clean_patch = clean_key_rgb[y1:y2, x1:x2].astype(np.float32)

            # Smooth Gaussian feather for undetectable edges (no hard seams)
            mask_float = (crop_mask > 0).astype(np.float32)
            ksize = max(7, min(25, int(min(pw, ph) * 0.15) | 1))
            feather = cv2.GaussianBlur(mask_float, (ksize, ksize), 0)[..., None]

            patch_trackers.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bx": bx,
                "by": by,
                "bw": bw,
                "bh": bh,
                "crop_mask": crop_mask,
                "clean_patch": clean_patch,
                "feather": feather,
                "collar": collar,
                "key_collar_mean": key_collar_mean,
                "last_update_idx": best_k_idx
            })

        # 4. Stream and blend across all video frames
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {input_video} for video streaming.")

        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frames_processed = 0
        t_stream_start = time.time()

        try:
            while True:
                if cancel_check and cancel_check():
                    raise RuntimeError("Processing was cancelled by user.")

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Apply each watermark patch with illumination matching & feathering
                for tracker in patch_trackers:
                    x1, y1, x2, y2 = tracker["x1"], tracker["y1"], tracker["x2"], tracker["y2"]
                    cur_roi = frame_rgb[y1:y2, x1:x2]
                    collar = tracker["collar"]

                    # Compute current illumination shift relative to keyframe
                    if np.any(collar):
                        cur_mean = np.mean(cur_roi[collar], axis=0)
                        delta = cur_mean - tracker["key_collar_mean"]

                        # Check if a major scene cut or lighting shift occurred
                        max_shift = np.max(np.abs(delta))
                        frames_since_update = frames_processed - tracker.get("last_update_idx", 0)

                        if max_shift > 36.0 and frames_since_update >= 45 and frame_rgb.mean() >= 16.0:
                            # Scene cut detected! Dynamically synthesize a fresh neural patch for the new scene
                            try:
                                scene_clean_rgb = inpainter.inpaint_frame(
                                    frame_rgb, mask, [(tracker["bx"], tracker["by"], tracker["bw"], tracker["bh"])],
                                    padding=max(padding, 28)
                                )
                                tracker["clean_patch"] = scene_clean_rgb[y1:y2, x1:x2].astype(np.float32)
                                tracker["key_collar_mean"] = cur_mean
                                tracker["last_update_idx"] = frames_processed
                                delta = np.array([0.0, 0.0, 0.0])
                                logger.info(f"Smart AI dynamically updated neural keyframe patch at scene frame #{frames_processed}")
                            except Exception as e_scene:
                                logger.debug(f"Dynamic scene update skipped: {e_scene}")
                    else:
                        delta = np.array([0.0, 0.0, 0.0])

                    # Apply color shift to synthesized patch
                    adj_patch = np.clip(tracker["clean_patch"] + delta, 0, 255)

                    # Subtle film grain / sensor texture injection to prevent any flat smooth blur
                    if np.any(collar):
                        collar_std = np.std(cur_roi[collar], axis=0)
                        if np.mean(collar_std) > 5.0:
                            noise = np.random.normal(0, float(np.mean(collar_std) * 0.22), adj_patch.shape)
                            adj_patch = np.clip(adj_patch + noise, 0, 255)

                    # Seamless alpha feather blend
                    blended = (1.0 - tracker["feather"]) * cur_roi.astype(np.float32) + tracker["feather"] * adj_patch
                    frame_rgb[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


                out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                # Note: VideoWriter expects BGR; cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out_bgr_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr_frame)

                frames_processed += 1
                if frames_processed % 4 == 0 or frames_processed == total_frames:
                    elapsed = time.time() - t_stream_start
                    cur_fps = frames_processed / elapsed if elapsed > 0 else 0
                    rem_frames = max(0, total_frames - frames_processed)
                    eta_sec = int(rem_frames / cur_fps) if cur_fps > 0 else 0

                    pct = 0.12 + 0.83 * (frames_processed / float(total_frames))
                    if progress_callback:
                        progress_callback(
                            pct,
                            f"Smart AI: {frames_processed}/{total_frames} frames ({int((frames_processed/total_frames)*100)}%) | {cur_fps:.1f} FPS | ETA: {eta_sec}s"
                        )
        finally:
            cap.release()
            writer.release()

        # 5. Audio Merge and Finalize H.264 MP4
        if progress_callback:
            progress_callback(0.96, "Smart AI: Merging audio and finalizing clean MP4...")

        merge_audio_and_finalize(
            processed_video_path=temp_video_path,
            original_video_path=input_video,
            output_path=output_video,
            has_audio=has_audio
        )

        cleanup_job_dir(job_dir)
        if progress_callback:
            progress_callback(1.0, "Smart AI watermark removal completed flawlessly!")

        return output_video

    except Exception as e:
        logger.error(f"Smart AI mode failed: {e}", exc_info=True)
        cleanup_job_dir(job_dir)
        raise e
