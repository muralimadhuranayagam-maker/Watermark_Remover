"""
Quick Mode removal using FFmpeg delogo filter directly on the video stream.
"""
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from core.video_io import _ensure_ffmpeg_in_path, get_video_metadata

logger = logging.getLogger(__name__)


def run_quick_mode(
    input_video: Path,
    rectangles: List[Tuple[int, int, int, int]],
    output_video: Path,
    has_audio: Optional[bool] = None,
    total_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_flag: Optional[threading.Event] = None
) -> Path:
    """
    Remove watermarks using FFmpeg delogo filter in a single pass without extracting frames.
    Preserves audio with -c:a copy.
    """
    _ensure_ffmpeg_in_path()
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if not rectangles:
        raise ValueError("No watermark rectangles specified for Quick Mode.")

    # Get metadata if total_frames or has_audio not passed
    meta = get_video_metadata(input_video)
    width, height = meta["width"], meta["height"]
    if has_audio is None:
        has_audio = meta["has_audio"]
    if total_frames is None or total_frames <= 0:
        total_frames = meta["total_frames"]

    # Build delogo filter chain
    delogo_filters = []
    for (x, y, w, h) in rectangles:
        # Clamp to bounds
        clamped_x = max(0, min(x, width - 2))
        clamped_y = max(0, min(y, height - 2))
        clamped_w = max(2, min(w, width - clamped_x))
        clamped_h = max(2, min(h, height - clamped_y))
        delogo_filters.append(f"delogo=x={clamped_x}:y={clamped_y}:w={clamped_w}:h={clamped_h}:show=0")

    filter_complex = ",".join(delogo_filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p"
    ]

    if has_audio:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.append("-an")

    cmd.extend([
        "-nostats",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        str(output_video)
    ])

    logger.info(f"Quick Mode command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True
    )

    frame_pattern = re.compile(r"^frame=\s*(\d+)")
    all_output_lines = []

    try:
        while True:
            if cancel_flag is not None and cancel_flag.is_set():
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise RuntimeError("Processing was cancelled by user.")

            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue

            line_clean = line.strip()
            all_output_lines.append(line_clean)
            if len(all_output_lines) > 200:
                all_output_lines.pop(0)

            match = frame_pattern.match(line_clean)
            if match and total_frames > 0:
                current_frame = int(match.group(1))
                progress_pct = min(1.0, current_frame / float(total_frames))
                if progress_callback:
                    progress_callback(
                        progress_pct,
                        f"Quick Mode: processed frame {current_frame}/{total_frames} ({int(progress_pct * 100)}%)"
                    )

        process.wait()
        if process.returncode != 0:
            err_msg = "\n".join(all_output_lines[-25:])
            raise RuntimeError(f"FFmpeg delogo failed (code {process.returncode}): {err_msg}")

    finally:
        if process.poll() is None:
            process.kill()

    if progress_callback:
        progress_callback(1.0, "Quick Mode completed.")

    return output_video

