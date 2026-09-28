# Project: DeepFeik — Real-Time Webcam Face-Swapping Desktop Application

## Architecture

DeepFeik is architected around a strict 4-tier decoupled pipeline ensuring high real-time CPU throughput (>= 15 FPS, benchmarked at 30-50 FPS), clean testability via synthetic video streams, and full portability for future Android deployment.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Layer 4: Desktop Presentation Layer                  │
│       PyQt5 GUI (MainWindow, VideoWidget, SourceFacePanel,             │
│                  RecordControls, StatusBar, Shortcuts [Ctrl+V])        │
│       CLI Entrypoint (deepfeik.cli:main)                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Qt Signals & Slots
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Layer 3: Video Pipeline Subsystem                   │
│   - PipelineController (orchestrates capture, engine, and recorder)    │
│   - IVideoSource Interface (WebcamSource [DirectShow], MockCameraSource)
│   - Single-Slot Drop-Oldest Buffer (LatestFrameBuffer: zero lag)       │
│   - AsyncVideoRecorder (background worker, mp4v/XVID, wall-clock pacing)
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pure NumPy Frames
┌───────────────────────────────────▼────────────────────────────────────┐
│                  Layer 2: Core Face Swap Engine                        │
│   - FaceMeshTracker (MediaPipe FaceMesh, 468 landmarks, video mode)   │
│   - CanonicalMeshTopology (~130 landmark subset for stable Delaunay)   │
│   - PiecewiseAffineWarper (Affine warp per Delaunay triangle on ROI)   │
│   - ReinhardColorMatcher (Mean-Std LAB color matching on face mask)    │
│   - FeatheredAlphaBlender (Face oval mask, Gaussian blur feathering)   │
│   - FaceSwapEngine (high-level API: set_source_face, process_frame)   │
│   * ZERO GUI imports (no Qt, no Tkinter)                               │
│   * ZERO Hardware capture imports (no VideoCapture)                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Mathematical Primitives
┌───────────────────────────────────▼────────────────────────────────────┐
│                   Layer 1: Math & Geometry Utilities                   │
│   - Delaunay triangulation indices & static topology cache             │
│   - Coordinate transforms, Bounding Box ROI clipping                   │
│   - Color space converters & alpha compositing kernels                 │
└────────────────────────────────────────────────────────────────────────┘
```

## Feature Inventory

Every feature derived from the Phase 0 survey and ORIGINAL_REQUEST.md is cataloged below with its assigned milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1. Webcam Video Acquisition | Open webcam via `cv2.CAP_DSHOW` on Windows; async threaded acquisition | M1 | ORIGINAL_REQUEST §R1 |
| 2 | F2. Video Source Abstraction & Mock | `IVideoSource` interface with `MockCameraSource` for headless CI/testing & disconnect fallback | M1 | Survey / Test Infra |
| 3 | F3. Real-Time Face Detection & Landmarks | MediaPipe FaceMesh video tracking (468 landmarks, 8-14ms CPU latency) | M1 | ORIGINAL_REQUEST §R1 |
| 4 | F4. Source Image File Loading | File picker dialog supporting JPG, PNG, BMP, WebP images | M2 | ORIGINAL_REQUEST §R2 |
| 5 | F5. Clipboard Image Paste | Ctrl+V and button paste supporting bitmap pixels (RGBA/RGB) and Explorer file paths | M2 | ORIGINAL_REQUEST §R2 |
| 6 | F6. Face Validation & Error Handling | Detect face in source; reject images with no face; select primary face when multiple faces | M2 | ORIGINAL_REQUEST §R2 |
| 7 | F7. Delaunay Mesh Piecewise Affine Warping | Canonical ~130-landmark mesh warping with affine transforms per triangle (<8ms CPU) | M2 | ORIGINAL_REQUEST §R1 |
| 8 | F8. Skin Tone & Color Matching | Reinhard LAB Mean-Std transfer matching source face tone to user lighting (<2.5ms CPU) | M2 | ORIGINAL_REQUEST §R1 |
| 9 | F9. Seamless Edge Feathering & Blending | Face oval contour mask with Gaussian blur feathering & linear alpha composite (<1.5ms CPU) | M2 | ORIGINAL_REQUEST §R1 |
| 10 | F10. Real-Time CPU Pipeline Coordination | Core `FaceSwapEngine` delivering >=15 FPS on laptop CPU with pass-through when no face | M2 | ORIGINAL_REQUEST §R1 |
| 11 | F11. Live Preview GUI Display | PyQt5 MainWindow with low-latency (<1ms) video blit, responsive UI | M3 | ORIGINAL_REQUEST §R4 |
| 12 | F12. GUI Controls & Status Feedback | Load button, paste button (Ctrl+V), record button, status bar, and FPS counter | M3 | ORIGINAL_REQUEST §R4 |
| 13 | F13. Asynchronous Video Recording | `AsyncVideoRecorder` with bounded queue; mp4v (.mp4) and XVID (.avi) codecs | M3 | ORIGINAL_REQUEST §R3 |
| 14 | F14. Wall-Clock Paced Recording & Finalization | VideoWriter pacing at 30 FPS regardless of swap FPS; file save dialog; clean close | M3 | ORIGINAL_REQUEST §R3 |
| 15 | F15. Clean Application & Device Shutdown | Signal handlers and Qt `closeEvent` releasing camera and finalizing video without hang | M3 | Acceptance Criteria |
| 16 | F16. Pip-Installable Packaging | `pyproject.toml` configuration, dependencies, and CLI entry point (`deepfeik`) | M4 | ORIGINAL_REQUEST §R1 |
| 17 | F17. Android-Ready Architecture Validation | Verification of strict layer boundaries (Core Engine has zero GUI/hardware deps) | M4 | ORIGINAL_REQUEST §R1 |
| 18 | F18. CPU Performance Benchmarking | Automated benchmark asserting >= 15 FPS throughput and latency budgets on CPU | M4 | ORIGINAL_REQUEST §R1 |
| 19 | F19. 100% E2E Test Suite Pass (Tiers 1-4) | Passing all 72 E2E test cases across Feature, Boundary, Pairwise, and Workload tiers | M5 | Acceptance Criteria |
| 20 | F20. Adversarial Coverage Hardening (Tier 5) | White-box stress-testing, fault injection, and coverage closure | M5 | Acceptance Criteria |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Video Source & Landmark Engine | Features 1, 2, 3: `IVideoSource`, `WebcamSource`, `MockCameraSource`, `FaceMeshTracker`, canonical landmark topology | none | DONE |
| M2 | Face Warping, Blending & Core Swap Engine | Features 4, 5, 6, 7, 8, 9, 10: `PiecewiseAffineWarper`, `ReinhardColorMatcher`, `FeatheredAlphaBlender`, `FaceSwapEngine`, clipboard/file validation | M1 | IN_PROGRESS |
| M3 | Desktop GUI, Live Preview & Video Recording | Features 11, 12, 13, 14, 15: `MainWindow`, `VideoWidget`, `LatestFrameBuffer`, `AsyncVideoRecorder`, `PipelineController` | M2 | PLANNED |
| M4 | Packaging, CLI & Performance Benchmark | Features 16, 17, 18: `pyproject.toml`, `deepfeik.cli`, setup verification, automated CPU benchmark runner | M3 | PLANNED |
| M5 | Final Milestone: 100% E2E Pass & Adversarial Hardening | Features 19, 20: Run full 72-case E2E test suite (Tiers 1-4), fix any regressions, execute Tier 5 adversarial hardening | M4, TEST_READY | PLANNED |

In parallel with Milestones 1–4, the **E2E Testing Track** implements the opaque-box test infrastructure and all test cases across Tiers 1–4, culminating in `TEST_READY.md`.

## Interface Contracts

### 1. Video Source Interface (`deepfeik.pipeline.video_source`)
```python
class IVideoSource(ABC):
    @abstractmethod
    def open(self) -> bool: ...
    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ... # returns (ret, bgr_frame)
    @abstractmethod
    def release(self) -> None: ...
    @abstractmethod
    def is_opened(self) -> bool: ...
    @abstractmethod
    def get_fps(self) -> float: ...
    @abstractmethod
    def get_resolution(self) -> tuple[int, int]: ... # (width, height)
```

### 2. Core Face Swap Engine Interface (`deepfeik.core.engine`)
```python
class FaceSwapEngine:
    def set_source_face(self, image: np.ndarray) -> tuple[bool, str]:
        """Loads and processes source face from BGR image.
        Returns: (success: bool, message: str)
        If success is False, message explains failure (e.g. 'No face detected')."""
        ...

    def clear_source_face(self) -> None:
        """Clears active source face; process_frame reverts to passthrough."""
        ...

    def is_source_loaded(self) -> bool:
        """Returns True if a valid source face is currently loaded."""
        ...

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single BGR camera frame.
        If no source face or no face in frame, returns original frame unmodified.
        Otherwise returns seamlessly swapped BGR frame."""
        ...
```

### 3. Asynchronous Video Recorder Interface (`deepfeik.pipeline.recorder`)
```python
class AsyncVideoRecorder:
    def start_recording(self, output_path: str, width: int, height: int, target_fps: float = 30.0) -> bool:
        """Begins asynchronous recording to output_path (.mp4 or .avi)."""
        ...

    def record_frame(self, frame: np.ndarray) -> None:
        """Queues a frame for recording with wall-clock pacing."""
        ...

    def stop_recording(self) -> None:
        """Drains frame queue, closes VideoWriter cleanly, and finalizes file."""
        ...

    def is_recording(self) -> bool:
        """Returns True if recording is currently active."""
        ...
```

### 4. Clipboard Service Interface (`deepfeik.gui.clipboard`)
```python
class ClipboardService:
    @staticmethod
    def get_image_from_clipboard() -> tuple[Optional[np.ndarray], Optional[str]]:
        """Extracts image from system clipboard (bitmap or copied file).
        Returns: (bgr_image, error_message)"""
        ...
```

## Code Layout

```
deepfeik/
├── src/
│   └── deepfeik/
│       ├── __init__.py
│       ├── cli.py                        # CLI entrypoint
│       ├── core/                         # Core Face Swap Engine (zero GUI/capture deps)
│       │   ├── __init__.py
│       │   ├── face_mesh.py              # MediaPipe FaceMesh wrapper & landmark extraction
│       │   ├── landmarks.py              # Canonical landmark indices & topology constants
│       │   ├── warper.py                 # Piecewise affine Delaunay triangle warping
│       │   ├── color.py                  # Reinhard LAB Mean-Std color transfer
│       │   ├── blender.py                # Gaussian feathered mask blending
│       │   └── engine.py                 # High-level FaceSwapEngine API
│       ├── pipeline/                     # Video I/O & Pipeline management
│       │   ├── __init__.py
│       │   ├── video_source.py           # IVideoSource, WebcamSource (DirectShow)
│       │   ├── mock_camera.py            # MockCameraSource (for CI & fallback)
│       │   ├── frame_buffer.py           # LatestFrameBuffer (drop-oldest, zero latency)
│       │   ├── recorder.py               # AsyncVideoRecorder (wall-clock paced VideoWriter)
│       │   └── controller.py             # PipelineController connecting capture, engine, recorder
│       └── gui/                          # PyQt5 Desktop Presentation Layer
│           ├── __init__.py
│           ├── main_window.py            # Main application window & event wiring
│           ├── video_widget.py           # High-speed (<1ms) QImage rendering widget
│           ├── controls_widget.py        # Buttons: Load Image, Paste, Start/Stop Record
│           └── clipboard.py              # Pillow & Qt dual-mode clipboard extractor
├── tests/
│   ├── conftest.py                       # Fixtures: mock sources, test face images, headless Qt
│   ├── e2e/                              # 4-Tier E2E Test Suite
│   │   ├── test_tier1_features.py        # 24 tests: isolated feature coverage
│   │   ├── test_tier2_boundaries.py      # 24 tests: boundary & corner cases
│   │   ├── test_tier3_pairwise.py        # 18 tests: cross-feature combinations
│   │   └── test_tier4_scenarios.py       # 6 tests: realistic end-to-end sessions
│   ├── unit/                             # Unit tests for core modules
│   │   ├── test_face_mesh.py
│   │   ├── test_warper.py
│   │   ├── test_color.py
│   │   ├── test_blender.py
│   │   ├── test_engine.py
│   │   ├── test_camera.py
│   │   ├── test_recorder.py
│   │   └── test_clipboard.py
│   └── benchmarks/
│       └── test_benchmark_cpu.py         # Automated CPU FPS & latency verification
├── pyproject.toml                        # Build & dependency metadata
├── README.md
├── PROJECT.md                            # Global project index & specifications
├── TEST_INFRA.md                         # E2E test plan & specifications
└── TEST_READY.md                         # Signal of test suite readiness (published by Test Track)
```
