"""
Video I/O module for metadata extraction, frame reading, and FFmpeg stream merging.
"""
import json
import logging
import math
import os
import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TEMP_BASE_DIR = Path("C:/wmtool/temp")


def _ensure_ffmpeg_in_path() -> None:
    """Ensure ffmpeg/ffprobe are available in os.environ['PATH'] on Windows."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    # Check Winget default Gyan.FFmpeg path
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_pkgs = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.exists():
            for p in winget_pkgs.glob("Gyan.FFmpeg*/**/bin"):
                if (p / "ffmpeg.exe").exists():
                    os.environ["PATH"] = f"{p};{os.environ['PATH']}"
                    logger.info(f"Added FFmpeg to PATH: {p}")
                    return

    # Also try reading registry user/machine path
    try:
        import winreg
        paths = []
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(root, r"Environment") as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    paths.append(val)
            except Exception:
                pass
        combined = ";".join(paths)
        if combined:
            os.environ["PATH"] = f"{combined};{os.environ['PATH']}"
    except Exception:
        pass


_ensure_ffmpeg_in_path()



def ensure_temp_dir() -> Path:
    """Ensure base temp directory exists and return it."""
    try:
        TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
        return TEMP_BASE_DIR
    except Exception as e:
        logger.warning(f"Failed to create {TEMP_BASE_DIR}, falling back to local temp: {e}")
        fallback = Path("./temp")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def create_job_dir(job_id: Optional[str] = None) -> Path:
    """Create and return a unique temporary job directory."""
    if not job_id:
        job_id = uuid.uuid4().hex[:12]
    base = ensure_temp_dir()
    job_dir = base / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def cleanup_job_dir(job_dir: Path) -> None:
    """Safely delete the job temporary directory."""
    try:
        if job_dir.exists() and job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Error cleaning up {job_dir}: {e}")


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe and fallback to OpenCV.
    Returns:
        dict with keys: width, height, fps, duration, total_frames, has_audio, codec_name
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Use ffprobe via subprocess list with utf-8 encoding and error replacement
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        data = json.loads(res.stdout)
    except Exception as e:
        logger.warning(f"ffprobe execution failed, falling back to OpenCV: {e}")
        return _get_metadata_opencv(video_path)

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream = None
    has_audio = False

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = s
        elif codec_type == "audio":
            has_audio = True

    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec_name = video_stream.get("codec_name", "unknown")

    # Parse fps
    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
    try:
        fps_fraction = Fraction(fps_str)
        fps = float(fps_fraction) if fps_fraction.denominator != 0 else 30.0
    except Exception:
        fps = 30.0
    if fps <= 0 or math.isnan(fps):
        fps = 30.0

    # Parse duration
    duration = 0.0
    try:
        if "duration" in video_stream and video_stream["duration"] is not None:
            duration = float(video_stream["duration"])
        elif "duration" in format_info and format_info["duration"] is not None:
            duration = float(format_info["duration"])
    except Exception:
        duration = 0.0

    # Parse total frames
    total_frames = 0
    nb_frames_str = video_stream.get("nb_frames")
    if nb_frames_str and nb_frames_str.isdigit():
        total_frames = int(nb_frames_str)
    elif duration > 0 and fps > 0:
        total_frames = int(round(duration * fps))

    # If width/height or total_frames still zero, fallback to OpenCV
    if width <= 0 or height <= 0 or total_frames <= 0:
        cv_meta = _get_metadata_opencv(video_path)
        if width <= 0:
            width = cv_meta["width"]
        if height <= 0:
            height = cv_meta["height"]
        if total_frames <= 0:
            total_frames = cv_meta["total_frames"]
        if duration <= 0:
            duration = cv_meta["duration"]

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": max(total_frames, 1),
        "has_audio": has_audio,
        "codec_name": codec_name,
        "path": video_path
    }


def _get_metadata_opencv(video_path: Path) -> Dict[str, Any]:
    """Fallback metadata reader using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": max(total_frames, 1),
        "has_audio": False,
        "codec_name": "unknown",
        "path": video_path
    }


def extract_frame_rgb(video_path: Path, frame_idx: int = 0) -> np.ndarray:
    """
    Extract a single frame from video at frame_idx as an RGB numpy array (H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        frame_idx = max(0, min(frame_idx, total_frames - 1))
    else:
        frame_idx = max(0, frame_idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame at index {frame_idx} from {video_path}")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def merge_audio_and_finalize(
    processed_video_path: Path,
    original_video_path: Path,
    output_path: Path,
    has_audio: bool
) -> Path:
    """
    Re-encode processed video with H.264 (-crf 18 -preset fast -pix_fmt yuv420p)
    and copy original audio track if present.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if has_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(processed_video_path),
            "-i", str(original_video_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",  # ? means map audio if present, without failing if missing
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(processed_video_path),
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]

    logger.info(f"Running finalize command: {' '.join(cmd)}")
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg finalize error ({res.returncode}): {res.stderr}")

    return output_path
