import logging
import multiprocessing as mp
import os
import shutil
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageFile

# Enable loading truncated images so Gradio ImageEditor handles large/streaming images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

from core.auto_detect import auto_detect_watermarks
from core.lama_mode import get_lama_inpainter
from core.masks import build_mask_from_editor, extract_bounding_boxes, overlay_mask_on_frame
from core.pipeline import preview_single_frame, process_video
from core.video_io import (
    _ensure_ffmpeg_in_path,
    cleanup_job_dir,
    ensure_temp_dir,
    extract_frame_rgb,
    get_video_metadata,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WatermarkRemoverApp")

_ensure_ffmpeg_in_path()
ensure_temp_dir()

# Global state for active processing job and cancellation
_ACTIVE_CANCEL_EVENT = threading.Event()
_IS_PROCESSING = False

CUSTOM_CSS = """
/* Modern Dark Glassmorphic Theme */
:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --accent-color: #06b6d4;
    --bg-dark: #0f172a;
    --card-bg: rgba(30, 41, 59, 0.7);
    --border-color: rgba(255, 255, 255, 0.1);
}

body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #f1f5f9 !important;
}

.app-header {
    text-align: center;
    padding: 24px 16px 16px 16px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border-radius: 16px;
    border: 1px solid var(--border-color);
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

.app-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0 !important;
}

.app-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
}

.system-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-gpu {
    background: rgba(129, 140, 248, 0.15);
    color: #818cf8;
    border-color: rgba(129, 140, 248, 0.3);
}

.badge-local {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.3);
}

.card-panel {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    backdrop-filter: blur(12px) !important;
}

.btn-primary {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
}

.btn-ai {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4) !important;
    transition: all 0.2s ease !important;
}

.btn-ai:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.6) !important;
}

.btn-secondary {
    background: rgba(51, 65, 85, 0.8) !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.btn-danger {
    background: linear-gradient(135deg, #dc2626, #ef4444) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
}

.status-box {
    padding: 12px 16px;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-color);
    font-size: 0.92rem;
    line-height: 1.5;
}
"""


def get_hardware_info() -> str:
    """Return hardware execution provider status."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            return "DirectML (GPU) available / CPU Fallback ready"
        return "CPU Execution Provider"
    except Exception:
        return "CPU Execution"


def handle_video_upload(video_file: Optional[str]) -> Tuple[Any, Any, str, int]:
    """
    Called when a video is uploaded. Reads metadata, extracts frame 0,
    and updates slider range and image editor.
    """
    if not video_file:
        return (
            gr.update(value=None),
            gr.update(maximum=100, value=0),
            "Please upload a video file to begin.",
            0
        )

    try:
        video_path = Path(video_file)
        meta = get_video_metadata(video_path)
        total_frames = meta["total_frames"]
        width = meta["width"]
        height = meta["height"]
        fps = meta["fps"]
        duration = meta["duration"]
        has_audio = meta["has_audio"]

        frame_0 = extract_frame_rgb(video_path, 0)

        audio_str = "🔊 Yes" if has_audio else "🔇 No"
        info_text = (
            f"**Resolution**: {width}x{height} | "
            f"**FPS**: {fps:.2f} | "
            f"**Duration**: {duration:.2f}s ({total_frames} frames) | "
            f"**Audio**: {audio_str}"
        )

        max_slider = max(0, total_frames - 1)
        editor_value = {"background": frame_0, "layers": [], "composite": None}

        return (
            editor_value,
            gr.update(maximum=max_slider, value=0, interactive=True),
            info_text,
            total_frames
        )
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        return (
            gr.update(value=None),
            gr.update(maximum=100, value=0, interactive=False),
            f"⚠️ Error reading video: {str(e)}",
            0
        )


def handle_frame_change(video_file: Optional[str], frame_idx: int) -> Any:
    """Load and display the selected frame from the video in the editor."""
    if not video_file:
        return gr.update()
    try:
        video_path = Path(video_file)
        frame_rgb = extract_frame_rgb(video_path, int(frame_idx))
        return {"background": frame_rgb, "layers": [], "composite": None}
    except Exception as e:
        logger.warning(f"Error extracting frame {frame_idx}: {e}")
        return gr.update()


def handle_preview(
    video_file: Optional[str],
    editor_data: Any,
    frame_idx: int,
    mode: str,
    standard_method: str,
    dilation_px: int,
    padding_px: int,
    refine_mask: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Generate a side-by-side preview of the selected frame before and after removal.
    """
    if not video_file:
        return None, None, "⚠️ Please upload a video first."

    try:
        video_path = Path(video_file)
        meta = get_video_metadata(video_path)
        h, w = meta["height"], meta["width"]

        # Build mask from editor strokes
        mask = build_mask_from_editor(editor_data, target_shape=(h, w), dilation_px=dilation_px)
        boxes = extract_bounding_boxes(mask)

        if not boxes and not np.any(mask > 0):
            return None, None, "⚠️ Please draw over the watermark area on the frame before previewing."

        before_rgb, after_rgb, info_text = preview_single_frame(
            video_path=video_path,
            frame_idx=int(frame_idx),
            mask=mask,
            boxes=boxes,
            mode=mode,
            standard_method=standard_method,
            padding=int(padding_px),
            refine_mask=refine_mask
        )

        detected_count = len(boxes)
        status_msg = f"✅ Preview generated for frame #{frame_idx} ({detected_count} watermark region{'s' if detected_count > 1 else ''} detected). {info_text}"
        return before_rgb, after_rgb, status_msg

    except Exception as e:
        logger.error(f"Preview error: {e}", exc_info=True)
        return None, None, f"⚠️ Preview failed: {str(e)}"



def handle_show_mask(
    video_file: Optional[str],
    editor_data: Any,
    frame_idx: int,
    dilation_px: int
) -> Tuple[Optional[np.ndarray], str]:
    """Visualize the detected mask overlay and bounding boxes on the current frame."""
    if not video_file:
        return None, "⚠️ Please upload a video first."

    try:
        video_path = Path(video_file)
        meta = get_video_metadata(video_path)
        h, w = meta["height"], meta["width"]

        frame_rgb = extract_frame_rgb(video_path, int(frame_idx))
        mask = build_mask_from_editor(editor_data, target_shape=(h, w), dilation_px=dilation_px)
        boxes = extract_bounding_boxes(mask)

        if not boxes and not np.any(mask > 0):
            return frame_rgb, "⚠️ No mask drawn. Use the brush to draw over the watermark."

        overlay = overlay_mask_on_frame(frame_rgb, mask)
        return overlay, f"✅ Mask detected: {len(boxes)} bounding box(es) found with {dilation_px}px dilation."
    except Exception as e:
        return None, f"⚠️ Error building mask: {str(e)}"


def handle_auto_detect(
    video_file: Optional[str],
    frame_idx: int
) -> Tuple[Any, int, int, Optional[np.ndarray], str]:
    """
    Automatically detect watermarks across the video, update the ImageEditor
    with visual bounding layers, auto-configure dilation and padding sliders,
    and update mask preview overlay.
    """
    if not video_file:
        return gr.update(), 4, 25, None, "⚠️ Please upload a video file first."

    try:
        video_path = Path(video_file)
        meta = get_video_metadata(video_path)
        h, w = meta["height"], meta["width"]

        mask, boxes, auto_dilation, auto_padding = auto_detect_watermarks(video_path)
        frame_rgb = extract_frame_rgb(video_path, int(frame_idx))

        if not boxes:
            return gr.update(), auto_dilation, auto_padding, frame_rgb, "⚠️ No static watermark detected automatically. You can draw manually with the brush."

        # Create RGBA layer overlay for Gradio ImageEditor
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        for (bx, by, bw, bh) in boxes:
            layer[by:by+bh, bx:bx+bw] = [255, 0, 50, 190]

        editor_update = {
            "background": frame_rgb,
            "layers": [layer],
            "composite": None
        }

        overlay = overlay_mask_on_frame(frame_rgb, mask)
        status_msg = (
            f"✨ Auto-detected {len(boxes)} watermark region(s)! "
            f"Dilation automatically set to {auto_dilation}px, Padding set to {auto_padding}px. "
            f"Ready for preview or removal."
        )
        return editor_update, auto_dilation, auto_padding, overlay, status_msg

    except Exception as e:
        logger.error(f"Auto-detection error: {e}", exc_info=True)
        return gr.update(), 4, 25, None, f"⚠️ Auto-detection failed: {str(e)}"


def handle_auto_remove(
    video_file: Optional[str],
    frame_idx: int = 0,
    refine_mask: bool = True,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray], str]:
    """
    1-Click Auto Remove:
    Automatically detects watermarks, sizes dilation/padding, runs Smart AI,
    and returns the cleaned video with zero blur.
    """
    global _ACTIVE_CANCEL_EVENT, _IS_PROCESSING
    _ACTIVE_CANCEL_EVENT.clear()
    _IS_PROCESSING = True

    if not video_file:
        _IS_PROCESSING = False
        return None, None, None, "⚠️ Please upload a video file first."

    try:
        video_path = Path(video_file)
        progress(0.02, desc="✨ Auto-detecting watermarks in video...")

        mask, boxes, auto_dilation, auto_padding = auto_detect_watermarks(video_path)

        if not boxes and (mask is None or not np.any(mask > 0)):
            _IS_PROCESSING = False
            return None, None, None, "⚠️ Could not auto-detect watermarks. Please draw manually on the frame."

        ref_frame = extract_frame_rgb(video_path, int(frame_idx))
        overlay = overlay_mask_on_frame(ref_frame, mask)

        def progress_adapter(pct: float, desc: str):
            progress(pct, desc=desc)

        def cancel_check():
            return _ACTIVE_CANCEL_EVENT.is_set()

        output_video_path = process_video(
            input_video=video_path,
            mask=mask,
            rectangles=boxes,
            mode="Smart AI (Fast & Flawless - Zero Blur)",
            padding=int(auto_padding),
            reference_frame_idx=int(frame_idx),
            refine_mask=refine_mask,
            progress_callback=progress_adapter,
            cancel_check=cancel_check
        )

        _IS_PROCESSING = False
        res_path_str = str(output_video_path)
        return (
            res_path_str,
            res_path_str,
            overlay,
            f"🎉 1-Click Auto Removal Successful! Cleaned {len(boxes)} watermark(s) with Smart AI (zero blur). Saved: {output_video_path.name}"
        )

    except RuntimeError as re:
        _IS_PROCESSING = False
        if "cancelled" in str(re).lower():
            return None, None, None, "🛑 Auto removal cancelled by user."
        return None, None, None, f"⚠️ Auto removal runtime error: {str(re)}"
    except Exception as e:
        _IS_PROCESSING = False
        logger.error(f"1-Click Auto removal failed: {e}", exc_info=True)
        return None, None, None, f"⚠️ 1-Click Auto removal failed: {str(e)}"


def handle_process_video(
    video_file: Optional[str],
    editor_data: Any,
    frame_idx: int,
    mode: str,
    standard_method: str,
    dilation_px: int,
    padding_px: int,
    refine_mask: bool = True,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Execute full video watermark removal.
    """
    global _ACTIVE_CANCEL_EVENT, _IS_PROCESSING
    _ACTIVE_CANCEL_EVENT.clear()
    _IS_PROCESSING = True

    if not video_file:
        _IS_PROCESSING = False
        return None, None, "⚠️ Please upload a video file first."

    try:
        video_path = Path(video_file)
        meta = get_video_metadata(video_path)
        h, w = meta["height"], meta["width"]

        # Build dilated mask
        mask = build_mask_from_editor(editor_data, target_shape=(h, w), dilation_px=dilation_px)
        boxes = extract_bounding_boxes(mask)

        if not boxes and not np.any(mask > 0):
            _IS_PROCESSING = False
            return None, None, "⚠️ No watermark detected. Please draw over the watermark in the editor above."

        logger.info(f"Starting watermark removal: mode={mode}, boxes={boxes}")

        def progress_adapter(pct: float, desc: str):
            progress(pct, desc=desc)

        def cancel_check():
            return _ACTIVE_CANCEL_EVENT.is_set()

        output_video_path = process_video(
            input_video=video_path,
            mask=mask,
            rectangles=boxes,
            mode=mode,
            standard_method=standard_method,
            padding=int(padding_px),
            reference_frame_idx=int(frame_idx),
            refine_mask=refine_mask,
            progress_callback=progress_adapter,
            cancel_check=cancel_check
        )

        _IS_PROCESSING = False
        res_path_str = str(output_video_path)
        return (
            res_path_str,
            res_path_str,
            f"🎉 Success! Cleaned video saved: {output_video_path.name}"
        )

    except RuntimeError as re:
        _IS_PROCESSING = False
        if "cancelled" in str(re).lower():
            return None, None, "🛑 Processing was cancelled by user. Temporary files cleaned up."
        logger.error(f"Runtime error: {re}")
        return None, None, f"⚠️ Processing error: {str(re)}"
    except Exception as e:
        _IS_PROCESSING = False
        logger.error(f"Processing failed: {e}", exc_info=True)
        return None, None, f"⚠️ Failed to process video: {str(e)}"



def handle_cancel() -> str:
    """Cancel any active processing job."""
    global _ACTIVE_CANCEL_EVENT, _IS_PROCESSING
    if _IS_PROCESSING:
        _ACTIVE_CANCEL_EVENT.set()
        return "⏳ Cancelling active job and cleaning up resources..."
    return "No active job is currently running."


def create_ui() -> gr.Blocks:
    """Build the Gradio application UI."""
    hw_info = get_hardware_info()

    with gr.Blocks(title="Watermark Remover") as demo:
        # Header
        with gr.Row(elem_classes=["app-header"]):
            with gr.Column():
                gr.Markdown(
                    """
                    <div style="text-align: center;">
                        <h1 class="app-title">🎬 Video Watermark Remover</h1>
                        <p class="app-subtitle">Local Windows desktop tool for high-quality video watermark removal (Quick / OpenCV / LaMa ONNX)</p>
                        <div class="system-badges">
                            <span class="badge badge-local">🔒 100% Local & Offline</span>
                            <span class="badge badge-gpu">⚡ Hardware: DirectML / CPU Fallback</span>
                            <span class="badge">🎞️ Audio Copy & Fast H.264</span>
                        </div>
                    </div>
                    """
                )

        # Main Layout
        with gr.Row():
            # Left Column: Inputs & Settings
            with gr.Column(scale=5, elem_classes=["card-panel"]):
                gr.Markdown("### 1. Upload Video")
                video_input = gr.Video(
                    label="Upload Video (MP4, MOV, AVI, MKV, WEBM)",
                    sources=["upload"],
                    interactive=True
                )
                video_meta_info = gr.Markdown("Upload a video to inspect details.")
                total_frames_state = gr.State(value=0)

                gr.Markdown("### 2. Select Frame & Draw Watermark")
                with gr.Row():
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                        label="Use a different frame (if watermark is hidden on frame 0)",
                        interactive=False
                    )
                    refresh_frame_btn = gr.Button("🔄 Refresh Frame", size="sm")

                with gr.Row():
                    auto_detect_btn = gr.Button("✨ Auto-Detect Watermark", elem_classes=["btn-ai"], scale=1)
                    auto_remove_btn = gr.Button("⚡ 1-Click Auto Remove (Flawless)", elem_classes=["btn-primary"], scale=1)

                image_editor = gr.ImageEditor(
                    label="Draw rectangle or brush strokes over each watermark",
                    type="numpy",
                    brush=gr.Brush(default_size=24, colors=["#ff0033", "#00ffff", "#ffffff"]),
                    transforms=["crop"],
                    layers=True
                )

                with gr.Row():
                    dilation_slider = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        value=4,
                        label="Mask Dilation / Expansion (px) - covers fuzzy edges"
                    )
                    padding_slider = gr.Slider(
                        minimum=5,
                        maximum=60,
                        step=5,
                        value=25,
                        label="Cropped Inpaint Padding (px)"
                    )

                refine_mask_cb = gr.Checkbox(
                    value=True,
                    label="✨ Tighten Mask to Watermark Strokes (Zero Blur - Preserves Background)",
                    info="Isolates actual text and logo edges so real cobblestones, clothes, and road texture are never wiped out."
                )

                show_mask_btn = gr.Button("👁️ Verify Mask & Bounding Boxes", size="sm")


                gr.Markdown("### 3. Choose Removal Mode")
                mode_radio = gr.Radio(
                    choices=[
                        "Smart AI (Fast & Flawless - Zero Blur)",
                        "Standard (OpenCV inpainting)",
                        "High Quality (LaMa Frame-by-Frame)",
                        "Quick (FFmpeg delogo)"
                    ],
                    value="Smart AI (Fast & Flawless - Zero Blur)",
                    label="Inpainting Engine",
                    info="Select speed vs restoration fidelity"
                )

                with gr.Row(visible=False) as standard_options:
                    standard_algo = gr.Radio(
                        choices=["Telea (Fastest)", "Navier-Stokes (NS)"],
                        value="Telea (Fastest)",
                        label="Standard Inpainting Algorithm"
                    )

                mode_descriptions = gr.Markdown(
                    """
                    ✨ **Smart AI (Fast & Flawless - Zero Blur)**:
                    Recommended! Deep neural inpainting (LaMa ONNX) on keyframes with seamless illumination tracking across video frames at 60+ FPS. Synthesizes natural, sharp textures with **zero blur**!
                    """
                )

                def update_mode_desc(m: str):
                    if "Smart" in m:
                        return (
                            "✨ **Smart AI (Fast & Flawless - Zero Blur)**: Recommended! Uses deep neural network (LaMa ONNX) on keyframe to synthesize genuine natural background texture with **zero blur**, then streams with seamless illumination tracking at 60+ FPS.",
                            gr.update(visible=False)
                        )
                    elif "Quick" in m:
                        return (
                            "⚡ **Quick (FFmpeg delogo)**: Runs FFmpeg's native delogo filter directly on the video stream without frame extraction. Fastest possible (~200+ fps). Best for simple static corner watermarks.",
                            gr.update(visible=False)
                        )
                    elif "High" in m or "Frame-by-Frame" in m:
                        return (
                            "🧠 **High Quality (LaMa Frame-by-Frame)**: Deep learning inpainting model running on every frame sequentially via DirectML / CPU fallback. Highest accuracy for rapidly moving backgrounds under the watermark.",
                            gr.update(visible=False)
                        )
                    else:
                        return (
                            "🚀 **Standard (OpenCV inpainting)**: Inpaints cropped bounding box regions with OpenCV Telea algorithm. Blazing fast (~100+ fps) with multi-core parallel CPU workers.",
                            gr.update(visible=True)
                        )

                mode_radio.change(
                    fn=update_mode_desc,
                    inputs=[mode_radio],
                    outputs=[mode_descriptions, standard_options]
                )

                with gr.Row():
                    preview_btn = gr.Button("🔍 Preview One Frame", elem_classes=["btn-secondary"], scale=1)
                    process_btn = gr.Button("🚀 Remove Watermark", elem_classes=["btn-primary"], scale=2)
                    cancel_btn = gr.Button("🛑 Cancel", elem_classes=["btn-danger"], scale=1)

            # Right Column: Previews & Results
            with gr.Column(scale=5, elem_classes=["card-panel"]):
                gr.Markdown("### 4. Verification & Output")

                status_output = gr.Markdown(
                    "Ready. Upload a video and mark the watermark.",
                    elem_classes=["status-box"]
                )

                with gr.Tabs():
                    with gr.TabItem("🔍 Frame Preview"):
                        with gr.Row():
                            before_image = gr.Image(label="Original Frame", interactive=False)
                            after_image = gr.Image(label="Cleaned Frame", interactive=False)

                    with gr.TabItem("👁️ Mask & Detection Overlay"):
                        mask_preview_image = gr.Image(
                            label="Detected Watermark Mask & Bounding Rectangles",
                            interactive=False
                        )

                    with gr.TabItem("🎬 Final Video Result"):
                        video_output = gr.Video(label="Cleaned Video (H.264 + Original Audio)")
                        download_file = gr.File(label="Download Cleaned MP4")

        # Event Handlers
        video_input.change(
            fn=handle_video_upload,
            inputs=[video_input],
            outputs=[image_editor, frame_slider, video_meta_info, total_frames_state]
        )

        frame_slider.change(
            fn=handle_frame_change,
            inputs=[video_input, frame_slider],
            outputs=[image_editor]
        )

        refresh_frame_btn.click(
            fn=handle_frame_change,
            inputs=[video_input, frame_slider],
            outputs=[image_editor]
        )

        show_mask_btn.click(
            fn=handle_show_mask,
            inputs=[video_input, image_editor, frame_slider, dilation_slider],
            outputs=[mask_preview_image, status_output]
        )

        auto_detect_btn.click(
            fn=handle_auto_detect,
            inputs=[video_input, frame_slider],
            outputs=[image_editor, dilation_slider, padding_slider, mask_preview_image, status_output]
        )

        auto_remove_btn.click(
            fn=handle_auto_remove,
            inputs=[video_input, frame_slider, refine_mask_cb],
            outputs=[video_output, download_file, mask_preview_image, status_output]
        )

        def parse_standard_algo(algo_str: str) -> str:
            return "ns" if "navier" in algo_str.lower() else "telea"

        preview_btn.click(
            fn=lambda vid, ed, f_idx, m, algo, dil, pad, ref: handle_preview(
                vid, ed, f_idx, m, parse_standard_algo(algo), dil, pad, ref
            ),
            inputs=[
                video_input,
                image_editor,
                frame_slider,
                mode_radio,
                standard_algo,
                dilation_slider,
                padding_slider,
                refine_mask_cb
            ],
            outputs=[before_image, after_image, status_output]
        )

        process_btn.click(
            fn=lambda vid, ed, f_idx, m, algo, dil, pad, ref: handle_process_video(
                vid, ed, f_idx, m, parse_standard_algo(algo), dil, pad, ref
            ),
            inputs=[
                video_input,
                image_editor,
                frame_slider,
                mode_radio,
                standard_algo,
                dilation_slider,
                padding_slider,
                refine_mask_cb
            ],
            outputs=[video_output, download_file, status_output]
        )

        cancel_btn.click(
            fn=handle_cancel,
            inputs=[],
            outputs=[status_output]
        )

    return demo


def launch_browser():
    """Launch the browser automatically when server starts."""
    try:
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:7860")
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    mp.freeze_support()

    logger.info("Starting Video Watermark Remover at http://127.0.0.1:7860")
    threading.Thread(target=launch_browser, daemon=True).start()

    app = create_ui()
    outputs_dir = Path("./outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
        allowed_paths=["C:/wmtool/temp", "C:\\wmtool\\temp", str(outputs_dir)],
        show_error=True
    )
