# 🎬 Local Video Watermark Remover

A fast, private, 100% offline desktop application for Windows to remove watermarks, logos, and timestamps from videos without needing an NVIDIA GPU or cloud services.

Built with **Gradio**, **OpenCV**, **FFmpeg**, and **LaMa ONNX** with **DirectML** GPU acceleration and automatic CPU fallback.

---

## ✨ Features

- **100% Local & Private**: No video data ever leaves your computer. Runs completely offline on `http://127.0.0.1:7860`.
- **No NVIDIA GPU Required**: Fully optimized for AMD Ryzen CPUs and Radeon / Intel integrated graphics.
- **✨ 1-Click Auto Watermark Detection**:
  - Automatically identifies static logos, social handles (e.g. `@username`), timestamps, and text banners using temporal stability and edge frequency analysis.
  - Automatically computes optimal mask dilation (covering font fringes) and inpainting padding.
- **⚡ 1-Click Auto Remove**:
  - Upload a video and click one button: automatically detects watermarks, sizes dilation/padding, runs Smart AI, and exports the clean video with **zero blur**!
- **Four Inpainting Modes**:
  1. **✨ Smart AI (Fast & Flawless - Zero Blur)** *(Default & Recommended)*:
     Synthesizes clean background textures using the LaMa deep neural network on keyframes, then seamlessly tracks and blends the patch across all frames with local illumination matching at 60+ FPS. Total processing time is only ~10–20 seconds with **zero blur and perfect texture sync**!
  2. **Standard**: OpenCV cropped inpainting (Telea or Navier-Stokes) running in parallel across all CPU cores (`CPU cores - 1` workers). Blazing fast (~80–150 FPS).
  3. **High Quality**: Deep learning frame-by-frame inpainting with the LaMa ONNX model via DirectML / multi-threaded CPU fallback. Best for rapidly moving backgrounds under the watermark.
  4. **Quick**: Uses FFmpeg's native `delogo` filter directly on the video stream without frame extraction (~150–300+ FPS).
- **Audio & Stream Preservation**:
  - Automatically preserves original audio tracks with `-c:a copy` (lossless audio transfer).
  - Encodes with fast H.264 (`-crf 18 -preset fast -pix_fmt yuv420p`) compatible with Windows Media Player, phones, and web browsers.
- **Instant Single-Frame Preview**: Compare "Before" and "After" side-by-side on any frame before processing the entire video.
- **Progress & Cancellation**:
  - Real-time progress bar showing frames processed, FPS, and estimated time remaining (ETA).
  - Safe Cancel button that halts execution immediately and purges temporary files.

---

## 📋 System Requirements

- **Operating System**: Windows 10 or 11 (64-bit)
- **Python**: Python 3.11 (installed via winget or python.org)
- **FFmpeg**: Gyan.FFmpeg (installed via winget: `winget install Gyan.FFmpeg`)
- **Hardware**: Integrated graphics (AMD Radeon RX Vega, Intel UHD/Iris Xe) or standard multi-core CPU. 8GB+ RAM recommended.

---

## 🚀 Quick Start

### 1. Initial Setup
Run the setup script:
```cmd
setup.bat
```

### 2. Launch the Application
Simply double-click or run:
```cmd
start.bat
```
This activates the environment, starts the local server, and **automatically launches your default browser** at:
`http://127.0.0.1:7860`

---

## 🛠️ How to Use

### Automatic Workflow (Fastest & Easiest):
1. **Upload Video**: Drag and drop your video file.
2. **1-Click Auto Remove**: Click **⚡ 1-Click Auto Remove (Flawless)**.
3. The app automatically detects watermarks, sizes dilation/padding, runs Smart AI, and delivers the clean video with zero blur in seconds!

### Guided Workflow:
1. **Upload Video**: Upload your video file.
2. **Auto-Detect or Draw**: Click **✨ Auto-Detect Watermark** to automatically place bounding boxes, or use the brush tool in the editor.
3. **Verify Mask**: Click **👁️ Verify Mask & Bounding Boxes** to see the mask overlay.
4. **Choose Mode**: Keep the default **✨ Smart AI (Fast & Flawless - Zero Blur)** for best results.
5. **Check Preview**: Click **🔍 Preview One Frame** to inspect side-by-side before/after quality.
6. **Remove Watermark**: Click **🚀 Remove Watermark** to process the full video.
7. **Download**: Play the cleaned video or click download.

---

## 📁 Project Architecture

```
watermark-remover/
├── app.py                 # Gradio Web UI, event handlers, and browser launcher
├── core/
│   ├── __init__.py
│   ├── auto_detect.py     # Temporal variance & edge text watermark auto-detection
│   ├── smart_mode.py      # Smart AI: Keyframe LaMa inpainting + illumination feather blending
│   ├── lama_mode.py       # LaMa ONNX inpainter with DirectML & CPU fallback
│   ├── standard_mode.py   # OpenCV Telea & Navier-Stokes cropped inpainting
│   ├── quick_mode.py      # Native FFmpeg delogo stream processing
│   ├── masks.py           # Editor stroke extraction, dilation, bounding boxes, overlay
│   └── video_io.py        # ffprobe metadata, frame extraction, FFmpeg audio merge
├── models/
│   └── lama.onnx          # Pretrained LaMa inpainting ONNX model (~198MB)
├── requirements.txt       # Pinned Windows-compatible package versions
├── setup.bat              # One-click environment installer
├── start.bat              # One-click launcher
└── README.md
```

---

## 💡 Known Limitations & Tips

- **High Quality Mode (LaMa) Speed on CPU**:
  On integrated graphics where DirectML driver triggers a timeout (TDR), the app automatically falls back to CPU. AI inpainting on CPU processes ~1–5 frames/second. For longer videos (several minutes), **Standard Mode** is recommended as it achieves ~100+ FPS on standard multi-core processors.
- **Moving Watermarks**:
  The marked mask applies to all frames in the video. If the watermark moves across the screen (dynamic scrolling text), the mask should be drawn to cover the watermark's path or marked in segments.
- **Edge Coverage**:
  Use the **Mask Dilation** slider (default: 4px) to ensure faint translucent watermark halos and anti-aliased borders are completely covered before inpainting.
