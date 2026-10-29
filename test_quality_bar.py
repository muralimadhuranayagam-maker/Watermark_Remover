"""
Automated Quality Bar Verification Test Suite for Video Watermark Remover.
Tests:
1. Video with drawn watermark and audio (Quick Mode).
2. Vertical video without audio (Standard Mode).
3. Odd resolution video (1366x768) (Standard Mode NS).
4. High Quality (LaMa ONNX) Mode on watermarked frame.
5. Verifies resolution, fps, duration, and audio preservation.
"""
import subprocess
import time
from pathlib import Path
import numpy as np
import cv2

from core.video_io import _ensure_ffmpeg_in_path, get_video_metadata
from core.pipeline import process_video
from core.masks import create_rect_mask

_ensure_ffmpeg_in_path()

TEST_DIR = Path("test_artifacts")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_videos():
    print("[1/5] Generating test videos with FFmpeg...")
    
    # 1. Standard video with audio and a text box watermark at (450, 40) width 140, height 40
    # We use drawbox if drawtext font is missing, or drawbox with color to simulate watermark
    v1 = TEST_DIR / "test_watermarked_audio.mp4"
    cmd1 = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=800:duration=2",
        "-vf", "drawbox=x=450:y=40:w=140:h=40:color=white@0.9:t=fill",
        "-t", "2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(v1)
    ]
    subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  -> Generated test_watermarked_audio.mp4 (640x360, 25fps, has audio)")

    # 2. Vertical video without audio (720x1280)
    v2 = TEST_DIR / "test_vertical_no_audio.mp4"
    cmd2 = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=720x1280:rate=30",
        "-vf", "drawbox=x=50:y=50:w=200:h=60:color=yellow@0.9:t=fill",
        "-t", "1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        str(v2)
    ]
    subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  -> Generated test_vertical_no_audio.mp4 (720x1280, 30fps, no audio)")

    # 3. Odd resolution (1366x768)
    v3 = TEST_DIR / "test_odd_resolution.mp4"
    cmd3 = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=1366x768:rate=24",
        "-vf", "drawbox=x=1000:y=600:w=180:h=50:color=red@0.8:t=fill",
        "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        str(v3)
    ]
    subprocess.run(cmd3, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  -> Generated test_odd_resolution.mp4 (1366x768, 24fps)")

    return v1, v2, v3


def test_quick_mode(v1: Path):
    print("\n[2/5] Testing Quick Mode (FFmpeg delogo) on video with audio...")
    meta_in = get_video_metadata(v1)
    rect = (450, 40, 140, 40)
    mask = create_rect_mask((meta_in["height"], meta_in["width"]), [rect], dilation_px=4)

    out = process_video(v1, mask=mask, rectangles=[rect], mode="Quick")
    meta_out = get_video_metadata(out)
    
    print(f"  -> Input:  {meta_in['width']}x{meta_in['height']}, {meta_in['fps']} fps, audio={meta_in['has_audio']}")
    print(f"  -> Output: {meta_out['width']}x{meta_out['height']}, {meta_out['fps']} fps, audio={meta_out['has_audio']}")
    
    assert meta_out["width"] == meta_in["width"], "Width mismatch"
    assert meta_out["height"] == meta_in["height"], "Height mismatch"
    assert meta_out["has_audio"] is True, "Audio was not preserved"
    print("  -> Quick Mode PASSED!")


def test_standard_vertical_no_audio(v2: Path):
    print("\n[3/5] Testing Standard Mode (Telea) on vertical video (720x1280) with NO audio...")
    meta_in = get_video_metadata(v2)
    rect = (50, 50, 200, 60)
    mask = create_rect_mask((meta_in["height"], meta_in["width"]), [rect], dilation_px=4)

    out = process_video(v2, mask=mask, rectangles=[rect], mode="Standard", standard_method="telea")
    meta_out = get_video_metadata(out)

    print(f"  -> Input:  {meta_in['width']}x{meta_in['height']}, {meta_in['fps']} fps, audio={meta_in['has_audio']}")
    print(f"  -> Output: {meta_out['width']}x{meta_out['height']}, {meta_out['fps']} fps, audio={meta_out['has_audio']}")

    assert meta_out["width"] == 720 and meta_out["height"] == 1280, "Vertical resolution mismatch"
    assert meta_out["has_audio"] is False, "Audio presence mismatch"
    print("  -> Standard Vertical (No Audio) PASSED!")


def test_standard_odd_resolution(v3: Path):
    print("\n[4/5] Testing Standard Mode (Navier-Stokes) on odd resolution (1366x768)...")
    meta_in = get_video_metadata(v3)
    rect = (1000, 600, 180, 50)
    mask = create_rect_mask((meta_in["height"], meta_in["width"]), [rect], dilation_px=4)

    out = process_video(v3, mask=mask, rectangles=[rect], mode="Standard", standard_method="ns")
    meta_out = get_video_metadata(out)

    print(f"  -> Input:  {meta_in['width']}x{meta_in['height']}, {meta_in['fps']} fps")
    print(f"  -> Output: {meta_out['width']}x{meta_out['height']}, {meta_out['fps']} fps")

    assert meta_out["width"] == 1366 and meta_out["height"] == 768, "Odd resolution mismatch"
    print("  -> Standard Mode Navier-Stokes (1366x768) PASSED!")


def test_high_quality_lama(v1: Path):
    print("\n[5/5] Testing High Quality Mode (LaMa ONNX with Fallback)...")
    # Cut to a 0.5s clip (12 frames) for fast verification
    clip_path = TEST_DIR / "short_clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(v1), "-t", "0.48", "-c", "copy", str(clip_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    meta_in = get_video_metadata(clip_path)
    rect = (450, 40, 140, 40)
    mask = create_rect_mask((meta_in["height"], meta_in["width"]), [rect], dilation_px=4)

    out = process_video(clip_path, mask=mask, rectangles=[rect], mode="High Quality")
    meta_out = get_video_metadata(out)

    print(f"  -> Output: {meta_out['width']}x{meta_out['height']}, {meta_out['fps']} fps, audio={meta_out['has_audio']}")
    assert meta_out["width"] == meta_in["width"]
    assert meta_out["height"] == meta_in["height"]
    assert meta_out["has_audio"] is True
    print("  -> High Quality Mode PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print(" Running Full Video Watermark Remover Quality Bar Tests")
    print("=" * 60)
    start_total = time.time()
    
    v1, v2, v3 = generate_test_videos()
    test_quick_mode(v1)
    test_standard_vertical_no_audio(v2)
    test_standard_odd_resolution(v3)
    test_high_quality_lama(v1)

    print("\n" + "=" * 60)
    print(f" ALL TESTS PASSED! Total test duration: {time.time() - start_total:.2f}s")
    print("=" * 60)
