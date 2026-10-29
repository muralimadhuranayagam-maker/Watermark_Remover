"""
High Quality Mode: LaMa Inpainting ONNX with DirectML and CPU fallback.
"""
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests

logger = logging.getLogger(__name__)

LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama.onnx"
DEFAULT_MODEL_PATH = Path("models/lama.onnx")


def ensure_lama_model(
    model_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Path:
    """
    Ensure the LaMa ONNX model exists. If missing, download automatically with progress.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_path = Path(model_path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and model_path.stat().st_size > 50 * 1024 * 1024:
        return model_path

    logger.info(f"Downloading LaMa ONNX model to {model_path}...")
    if progress_callback:
        progress_callback(0.0, "Downloading LaMa ONNX model (~198MB)...")

    try:
        response = requests.get(LAMA_MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        pct = downloaded / total_size
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        progress_callback(pct, f"Downloading LaMa model: {mb_done:.1f}/{mb_total:.1f} MB ({int(pct*100)}%)")

        if progress_callback:
            progress_callback(1.0, "LaMa ONNX model downloaded successfully.")
        return model_path

    except Exception as e:
        logger.error(f"Failed to automatically download LaMa ONNX model: {e}")
        # Print exact manual instructions
        print("=" * 60)
        print("[ERROR] Automatic download of LaMa ONNX model failed.")
        print("Please manually download the model:")
        print(f"URL: {LAMA_MODEL_URL}")
        print(f"Save destination: {model_path}")
        print("=" * 60)
        raise RuntimeError(
            f"LaMa ONNX model download failed. Please manually download from {LAMA_MODEL_URL} and place it at {model_path}"
        )


_CACHED_INPAINTER: Optional["LamaInpainter"] = None
_DIRECTML_PREVIOUSLY_FAILED: bool = False


def get_lama_inpainter(
    model_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> "LamaInpainter":
    """Return cached singleton instance of LamaInpainter."""
    global _CACHED_INPAINTER
    if _CACHED_INPAINTER is None:
        _CACHED_INPAINTER = LamaInpainter(model_path, progress_callback)
    return _CACHED_INPAINTER


class LamaInpainter:
    """
    LaMa ONNX Inpainter supporting DirectML with automatic CPU fallback.
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self.model_path = ensure_lama_model(model_path, progress_callback)
        self.session = None
        self.active_provider = "Unknown"
        self._init_session()

    def _init_session(self) -> None:
        """Initialize ONNX session trying DirectML first, falling back to CPU."""
        global _DIRECTML_PREVIOUSLY_FAILED
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 1. Try DirectML first if not previously failed
        dml_available = ("DmlExecutionProvider" in ort.get_available_providers()) and not _DIRECTML_PREVIOUSLY_FAILED
        if dml_available:
            try:
                logger.info("Attempting to load LaMa with DmlExecutionProvider...")
                sess = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=opts,
                    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
                )
                # Verify DirectML with dummy inference to detect any GPU driver suspension early
                dummy_img = np.full((1, 3, 512, 512), 128.0, dtype=np.float32)
                dummy_mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
                sess.run(None, {"l_image_": dummy_img, "l_mask_": dummy_mask})

                self.session = sess
                self.active_provider = "DirectML (GPU Accelerated)"
                logger.info("LaMa initialized successfully with DirectML (GPU).")
                return
            except Exception as e:
                _DIRECTML_PREVIOUSLY_FAILED = True
                logger.warning(f"DirectML execution failed ({e}), falling back to CPUExecutionProvider.")

        # 2. Fallback to CPUExecutionProvider
        try:
            logger.info("Initializing LaMa with CPUExecutionProvider...")
            sess = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.session = sess
            self.active_provider = "CPU Execution Provider (CPU Fallback)"
            logger.info("LaMa initialized successfully with CPU fallback.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LaMa model session: {e}")


    def inpaint_crop(self, crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """
        Inpaint a cropped region using LaMa.
        crop_rgb: uint8 (H, W, 3) RGB
        crop_mask: uint8 (H, W), 255 for watermark
        """
        orig_h, orig_w = crop_rgb.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return crop_rgb

        # Resize to model expected size 512x512
        img_512 = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_512 = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Prepare float32 tensors with proper [0.0, 1.0] normalization
        # Note: LaMa expects image values scaled to [0.0, 1.0]
        img_tensor = np.transpose((img_512.astype(np.float32) / 255.0), (2, 0, 1))[None, ...]
        mask_tensor = (mask_512 > 0).astype(np.float32)[None, None, ...]

        # Run inference
        raw_out = self.session.run(None, {"l_image_": img_tensor, "l_mask_": mask_tensor})[0]

        # Post-process (output is already in [0, 255] float32)
        out_512 = np.clip(raw_out[0], 0, 255).astype(np.uint8)
        out_512 = np.transpose(out_512, (1, 2, 0))  # HWC RGB

        # Resize back to original crop resolution
        out_orig = cv2.resize(out_512, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Seamless alpha blending with smooth feathered edge
        mask_float = (crop_mask > 0).astype(np.float32)
        if orig_h > 8 and orig_w > 8:
            feather = cv2.GaussianBlur(mask_float, (7, 7), 0)[..., None]
        else:
            feather = mask_float[..., None]

        blended = (1.0 - feather) * crop_rgb.astype(np.float32) + feather * out_orig.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def inpaint_frame(
        self,
        frame_rgb: np.ndarray,
        mask: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        padding: int = 32
    ) -> np.ndarray:
        """
        Inpaint watermark regions on a full video frame in RGB format.
        Crops each box + padding, inpaints with LaMa, and pastes back.
        """
        if mask is None or not np.any(mask > 0):
            return frame_rgb

        out_frame = frame_rgb.copy()
        h, w = frame_rgb.shape[:2]

        if not boxes:
            from core.masks import extract_bounding_boxes
            boxes = extract_bounding_boxes(mask)

        if not boxes:
            return out_frame

        for (bx, by, bw, bh) in boxes:
            x1 = max(0, bx - padding)
            y1 = max(0, by - padding)
            x2 = min(w, bx + bw + padding)
            y2 = min(h, by + bh + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            crop_img = out_frame[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]

            if not np.any(crop_mask > 0):
                continue

            inpainted_crop = self.inpaint_crop(crop_img, crop_mask)
            out_frame[y1:y2, x1:x2] = inpainted_crop

        return out_frame
