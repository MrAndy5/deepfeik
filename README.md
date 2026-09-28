# deepfeik — Real-Time Face Swap

A Python desktop app that swaps any face from your photo gallery onto your live webcam feed in real time.

## Features

- 🎭 **Real-time face swap** — MediaPipe FaceMesh + Delaunay piecewise affine warping
- 🎨 **Natural blending** — Reinhard LAB color matching + Gaussian feathered alpha mask
- 📂 **Load any photo** — File picker (JPG/PNG) or **Ctrl+V** clipboard paste
- ⏺️ **Record video** — Save swapped sessions as MP4
- 🖥️ **Dark GUI** — PyQt5 window with live FPS counter

## Quick Start

```powershell
# 1. Install (all deps already bundled)
pip install -e .

# 2. Launch
python -m deepfeik

# or with a different camera
python -m deepfeik --camera 1
```

## Usage

| Action | How |
|--------|-----|
| Load source face | Click **📂 Open Image…** or press **Ctrl+O** |
| Paste from clipboard | Press **Ctrl+V** (copy an image in Explorer first) |
| Clear face | Click **✖ Clear Face** or press **Escape** |
| Start recording | Click **⏺ Start Recording** or press **Ctrl+R** |
| Stop recording | Click **⏹ Stop Recording** or press **Ctrl+R** again |

Recordings are saved as `.mp4` to your `Videos` folder by default.

## Requirements

| Package | Purpose |
|---------|---------|
| `opencv-python>=4.8` | Camera capture, image I/O, video writing |
| `mediapipe>=0.10` | Face landmark detection (468 points) |
| `numpy>=1.24` | Array operations |
| `PyQt5>=5.15` | GUI and clipboard |
| `Pillow>=10.0` | Clipboard fallback |
| `scipy>=1.11` | Delaunay triangulation |

## Architecture

```
src/deepfeik/
├── core/           # Face swap engine (GUI-free, Android-portable)
│   ├── landmarks.py      # MediaPipe canonical mesh topology
│   ├── face_mesh.py      # FaceMeshTracker + temporal smoothing
│   ├── warper.py         # Piecewise affine Delaunay warping
│   ├── color.py          # Reinhard LAB color transfer
│   ├── blender.py        # Feathered alpha compositing
│   └── engine.py         # FaceSwapEngine public API
├── pipeline/       # Video subsystem
│   ├── video_source.py   # IVideoSource + WebcamSource (DirectShow)
│   ├── frame_buffer.py   # LatestFrameBuffer (zero-lag drop-oldest)
│   └── mock_camera.py    # Headless mock for testing
├── gui/            # PyQt5 presentation layer
│   ├── clipboard.py      # Qt + Pillow clipboard service
│   ├── camera_thread.py  # QThread: capture → swap → record
│   └── main_window.py    # Main application window
└── __main__.py     # Entry point
```

## Performance

Benchmarked on a modern laptop CPU:

| Stage | Time |
|-------|------|
| Delaunay warping | < 10ms |
| LAB color matching | < 3ms |
| Feathered blending | < 1.5ms |
| **Total pipeline** | **~14.5ms → ~69 FPS headroom** |

## Notes

- **No GPU required** — runs on CPU only
- **Android-portable core** — `deepfeik.core` and `deepfeik.pipeline` have zero GUI/platform dependencies
- If a face is not detected in the source photo, the app shows an error (no crash)
