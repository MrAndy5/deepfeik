"""Global Pytest Fixtures and Mock Test Infrastructure for DeepFeik.

Provides:
- Headless Qt setup with QT_QPA_PLATFORM=offscreen
- MockCameraSource conforming to IVideoSource and supporting STATIC, PROCEDURAL, and SCRIPTED modes
- Synthesized test face images (valid face, blank/no-face, multi-face, corrupt, extreme aspect ratios, RGBA, grayscale)
- Import helpers for progressive testability against deepfeik components
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Ensure headless Qt offscreen platform is set before any Qt imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure src/ is on sys.path so deepfeik is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import pytest

# ============================================================================
# Synthetic Image Generation Utilities
# ============================================================================

def create_synthetic_face_image(
    width: int = 640,
    height: int = 480,
    cx: Optional[int] = None,
    cy: Optional[int] = None,
    axes: Tuple[int, int] = (100, 140),
    angle: float = 0.0,
    skin_color: Tuple[int, int, int] = (175, 195, 235),  # Peach BGR
    bg_color: Tuple[int, int, int] = (40, 40, 40),
    eye_open: float = 1.0,
    mouth_open: float = 0.5,
) -> np.ndarray:
    """Generates a synthetic frame containing a geometrically consistent face."""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    if cx is None:
        cx = width // 2
    if cy is None:
        cy = height // 2

    # Draw skin-colored face oval
    cv2.ellipse(frame, (cx, cy), axes, angle, 0, 360, skin_color, -1)
    # Face outline
    cv2.ellipse(frame, (cx, cy), axes, angle, 0, 360, (skin_color[0] - 30, skin_color[1] - 30, skin_color[2] - 30), 2)

    # Hair / scalp arc
    hair_axes = (axes[0] + 8, int(axes[1] * 0.75))
    hair_cy = cy - int(axes[1] * 0.35)
    cv2.ellipse(frame, (cx, hair_cy), hair_axes, angle, 180, 360, (25, 20, 20), -1)

    # Eyes & Eyebrows
    eye_offset_x = int(axes[0] * 0.42)
    eye_y = cy - int(axes[1] * 0.22)
    eye_rx = 14
    eye_ry = max(2, int(9 * eye_open))

    for sign in (-1, 1):
        ex = cx + sign * eye_offset_x
        # Eyebrows
        brow_y = eye_y - 18
        cv2.line(frame, (ex - 15, brow_y + 3), (ex + 15, brow_y - 3 if sign < 0 else brow_y + 3), (30, 25, 25), 3)
        # Sclera (white)
        cv2.ellipse(frame, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (250, 250, 250), -1)
        # Iris (blue/brown)
        cv2.circle(frame, (ex, eye_y), min(eye_ry, 6), (120, 70, 40), -1)
        # Pupil (black)
        cv2.circle(frame, (ex, eye_y), max(1, min(eye_ry // 2, 3)), (10, 10, 10), -1)

    # Nose
    nose_top = (cx, cy - int(axes[1] * 0.05))
    nose_bottom = (cx, cy + int(axes[1] * 0.18))
    cv2.line(frame, nose_top, nose_bottom, (140, 160, 200), 2)
    # Nostrils
    cv2.circle(frame, (cx - 8, nose_bottom[1]), 3, (120, 140, 180), -1)
    cv2.circle(frame, (cx + 8, nose_bottom[1]), 3, (120, 140, 180), -1)

    # Mouth & Lips
    mouth_y = cy + int(axes[1] * 0.45)
    mouth_rx = int(axes[0] * 0.35)
    mouth_ry = max(2, int(14 * mouth_open))
    # Lips
    cv2.ellipse(frame, (cx, mouth_y), (mouth_rx + 4, mouth_ry + 6), 0, 0, 360, (110, 100, 190), -1)
    # Inner mouth opening
    if mouth_open > 0.1:
        cv2.ellipse(frame, (cx, mouth_y), (mouth_rx, mouth_ry), 0, 0, 360, (40, 30, 70), -1)

    return frame


def create_blank_image(width: int = 640, height: int = 480, bg_color: Tuple[int, int, int] = (50, 60, 70)) -> np.ndarray:
    """Generates an image without any human facial features (gradient/texture)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / max(1, height - 1)
        img[y, :, 0] = int(bg_color[0] * (1.0 - ratio * 0.5))
        img[y, :, 1] = int(bg_color[1] * (1.0 - ratio * 0.3))
        img[y, :, 2] = int(bg_color[2] * (1.0 + ratio * 0.4))
    return img


def create_multi_face_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates an image containing two distinct faces:
    Primary face: centered at (200, 240) with larger area (axes: 95, 135)
    Secondary face: centered at (460, 220) with smaller area (axes: 55, 75)
    """
    frame = np.full((height, width, 3), (35, 35, 35), dtype=np.uint8)
    # Primary face (larger)
    face_primary = create_synthetic_face_image(
        width=width, height=height,
        cx=200, cy=240, axes=(95, 135),
        skin_color=(175, 195, 235),
        bg_color=(0, 0, 0)
    )
    # Secondary face (smaller)
    face_secondary = create_synthetic_face_image(
        width=width, height=height,
        cx=460, cy=220, axes=(55, 75),
        skin_color=(150, 175, 215),
        bg_color=(0, 0, 0)
    )
    # Composite onto frame
    mask_pri = (face_primary != 0).any(axis=2)
    mask_sec = (face_secondary != 0).any(axis=2)
    frame[mask_sec] = face_secondary[mask_sec]
    frame[mask_pri] = face_primary[mask_pri]
    return frame


def create_transparent_rgba_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a 4-channel RGBA image with a face and transparent background."""
    bgr = create_synthetic_face_image(width=width, height=height, bg_color=(0, 0, 0))
    alpha = np.zeros((height, width), dtype=np.uint8)
    face_mask = (bgr != 0).any(axis=2)
    alpha[face_mask] = 255
    bgra = np.dstack([bgr, alpha])
    return bgra


def create_grayscale_face_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a 1-channel Grayscale B&W face portrait."""
    bgr = create_synthetic_face_image(width=width, height=height)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


# ============================================================================
# Mock Camera Source (conforming to IVideoSource Interface)
# ============================================================================

class MockCameraSource:
    """
    Drop-in video source for automated headless testing and disconnect fallback.
    Conforms to deepfeik.pipeline.video_source.IVideoSource interface:
      - open() -> bool
      - read() -> tuple[bool, Optional[np.ndarray]]
      - release() -> None
      - is_opened() -> bool
      - get_fps() -> float
      - get_resolution() -> tuple[int, int]
    Supports cv2 duck-typing (isOpened, get, set).
    Modes:
      - 'STATIC': yields identical static face frame continuously
      - 'PROCEDURAL': generates dynamically animated moving face with blinking/mouth movement
      - 'VIDEO_FILE': reads from a video file path on disk
      - 'SCRIPTED': executes timeline scripts or fault injections (e.g. dropouts, disconnects)
    """

    def __init__(
        self,
        mode: str = "STATIC",
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        static_image: Optional[np.ndarray] = None,
        video_path: Optional[str] = None,
        timeline_script: Optional[List[Dict[str, Any]]] = None,
    ):
        self.mode = mode.upper()
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self._is_opened = True
        self.frame_idx = 0
        self.start_time = time.time()

        if static_image is not None:
            self.static_frame = cv2.resize(static_image, (self.width, self.height))
        else:
            self.static_frame = create_synthetic_face_image(width=self.width, height=self.height)

        self.video_path = video_path
        self._video_cap: Optional[cv2.VideoCapture] = None
        if self.mode == "VIDEO_FILE" and video_path:
            self._video_cap = cv2.VideoCapture(video_path)

        self.timeline_script = timeline_script or []
        self._fault_injections: Dict[int, str] = {}

    def open(self) -> bool:
        """Opens the camera stream."""
        self._is_opened = True
        return True

    def is_opened(self) -> bool:
        """Returns True if the camera stream is currently open."""
        return self._is_opened

    def isOpened(self) -> bool:
        """cv2 duck-typing compatibility."""
        return self.is_opened()

    def release(self) -> None:
        """Releases the camera stream resources."""
        self._is_opened = False
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

    def get_fps(self) -> float:
        """Returns current configured FPS."""
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        """Returns (width, height) resolution tuple."""
        return (self.width, self.height)

    def inject_fault(self, frame_num: int, fault_type: str) -> None:
        """Injects a fault at a specific frame index: 'disconnect', 'dropout', 'corrupt'."""
        self._fault_injections[frame_num] = fault_type

    def reset(self) -> None:
        """Resets stream frame index to 0."""
        self.frame_idx = 0
        self._is_opened = True
        if self._video_cap is not None:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set_resolution(self, width: int, height: int) -> None:
        """Updates stream resolution."""
        self.width = int(width)
        self.height = int(height)
        if self.static_frame is not None:
            self.static_frame = cv2.resize(self.static_frame, (self.width, self.height))

    def set_fps(self, fps: float) -> None:
        """Updates stream FPS."""
        self.fps = float(fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next video frame. Returns (ret, bgr_frame)."""
        if not self._is_opened:
            return False, None

        idx = self.frame_idx
        self.frame_idx += 1

        # Check injected faults
        if idx in self._fault_injections:
            fault = self._fault_injections[idx]
            if fault == "disconnect":
                self._is_opened = False
                return False, None
            elif fault == "dropout":
                return False, None
            elif fault == "corrupt":
                # Return noisy / corrupt frame
                return True, np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)

        # Mode 1: Static Frame Loop
        if self.mode == "STATIC":
            return True, self.static_frame.copy()

        # Mode 2: Procedural Animated Face
        elif self.mode == "PROCEDURAL":
            t = idx * 0.08
            cx = int(self.width // 2 + (self.width * 0.08) * np.sin(t))
            cy = int(self.height // 2 + (self.height * 0.05) * np.cos(t * 0.7))
            angle = float(4.0 * np.sin(t * 0.5))
            eye_open = 0.1 if (idx % 40) in (0, 1) else 1.0  # periodic blink
            mouth_open = 0.5 + 0.4 * np.sin(t * 1.5)  # talking motion

            frame = create_synthetic_face_image(
                width=self.width,
                height=self.height,
                cx=cx,
                cy=cy,
                axes=(int(self.width * 0.16), int(self.height * 0.28)),
                angle=angle,
                eye_open=eye_open,
                mouth_open=mouth_open,
            )
            return True, frame

        # Mode 3: Video File Loop
        elif self.mode == "VIDEO_FILE":
            if self._video_cap is None:
                return False, None
            ret, frame = self._video_cap.read()
            if not ret:
                # Loop back to frame 0
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._video_cap.read()
            if ret and frame is not None:
                return True, cv2.resize(frame, (self.width, self.height))
            return False, None

        # Mode 4: Scripted Timeline
        elif self.mode == "SCRIPTED":
            # Find matching segment in timeline_script
            for seg in self.timeline_script:
                start = seg.get("start", 0)
                end = seg.get("end", float("inf"))
                if start <= idx < end:
                    seg_type = seg.get("type", "normal")
                    if seg_type == "blank":
                        return True, create_blank_image(self.width, self.height)
                    elif seg_type == "multi":
                        return True, create_multi_face_image(self.width, self.height)
                    elif seg_type == "dropout":
                        return False, None
                    elif seg_type == "disconnect":
                        self._is_opened = False
                        return False, None
                    elif seg_type == "corrupt":
                        return True, np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
                    elif seg_type == "static":
                        return True, self.static_frame.copy()
            # Default to static frame if no segment matches
            return True, self.static_frame.copy()

        return False, None

    # cv2 duck-typing properties
    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.frame_idx)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self.set_resolution(int(value), self.height)
            return True
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self.set_resolution(self.width, int(value))
            return True
        elif prop_id == cv2.CAP_PROP_FPS:
            self.set_fps(float(value))
            return True
        return False


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def headless_qapp():
    """Session fixture creating offscreen PyQt5 application."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    yield app


@pytest.fixture
def mock_camera():
    """Yields a MockCameraSource in STATIC mode (640x480, 30 FPS)."""
    cam = MockCameraSource(mode="STATIC", fps=30.0, width=640, height=480)
    yield cam
    cam.release()


@pytest.fixture
def mock_animated_camera():
    """Yields a MockCameraSource in PROCEDURAL animated mode."""
    cam = MockCameraSource(mode="PROCEDURAL", fps=30.0, width=640, height=480)
    yield cam
    cam.release()


@pytest.fixture
def mock_scripted_camera():
    """Yields a MockCameraSource configured with a multi-phase test script."""
    script = [
        {"start": 0, "end": 15, "type": "normal"},       # 15 frames of face
        {"start": 15, "end": 30, "type": "blank"},       # 15 frames of no face
        {"start": 30, "end": 45, "type": "multi"},       # 15 frames of multi-face
        {"start": 45, "end": 55, "type": "dropout"},     # 10 frames of camera dropout
        {"start": 55, "end": 75, "type": "normal"},      # 20 frames resumed normal face
    ]
    cam = MockCameraSource(mode="SCRIPTED", fps=30.0, width=640, height=480, timeline_script=script)
    yield cam
    cam.release()


@pytest.fixture
def valid_face_image() -> np.ndarray:
    """Returns a BGR numpy array containing a valid synthetic face."""
    return create_synthetic_face_image(width=640, height=480)


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """Generates a synthetic 640x480 BGR image with facial features detectable by MediaPipe."""
    return create_synthetic_face_image(width=640, height=480)


@pytest.fixture
def blank_black_image() -> np.ndarray:
    """Generates an all-black 640x480 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_noise_image() -> np.ndarray:
    """Generates a random uniform noise 640x480 BGR image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Standard 640x480 BGR test frame."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def valid_face_path(tmp_path: Path, valid_face_image: np.ndarray) -> Path:
    """Saves valid face image as JPEG and returns file Path."""
    p = tmp_path / "valid_face.jpg"
    cv2.imwrite(str(p), valid_face_image)
    return p


@pytest.fixture
def valid_png_path(tmp_path: Path, valid_face_image: np.ndarray) -> Path:
    """Saves valid face image as PNG and returns file Path."""
    p = tmp_path / "valid_face.png"
    cv2.imwrite(str(p), valid_face_image)
    return p


@pytest.fixture
def blank_image() -> np.ndarray:
    """Returns a BGR numpy array with no face (gradient/scenery)."""
    return create_blank_image(width=640, height=480)


@pytest.fixture
def blank_image_path(tmp_path: Path, blank_image: np.ndarray) -> Path:
    """Saves blank image as JPEG and returns file Path."""
    p = tmp_path / "blank_image.jpg"
    cv2.imwrite(str(p), blank_image)
    return p


@pytest.fixture
def multi_face_image() -> np.ndarray:
    """Returns a BGR numpy array containing two faces (primary and secondary)."""
    return create_multi_face_image(width=640, height=480)


@pytest.fixture
def multi_face_path(tmp_path: Path, multi_face_image: np.ndarray) -> Path:
    """Saves multi-face image as JPEG and returns file Path."""
    p = tmp_path / "multi_face.jpg"
    cv2.imwrite(str(p), multi_face_image)
    return p


@pytest.fixture
def corrupt_image_path(tmp_path: Path) -> Path:
    """Creates a 0-byte corrupt file and returns file Path."""
    p = tmp_path / "corrupt_zero_byte.jpg"
    p.write_bytes(b"")
    return p


@pytest.fixture
def extreme_aspect_wide_path(tmp_path: Path) -> Path:
    """Creates a panoramic wide image (2000x200) and returns file Path."""
    p = tmp_path / "extreme_wide_2000x200.jpg"
    img = create_blank_image(width=2000, height=200)
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def extreme_aspect_tall_path(tmp_path: Path) -> Path:
    """Creates a very tall image (200x2000) and returns file Path."""
    p = tmp_path / "extreme_tall_200x2000.jpg"
    img = create_blank_image(width=200, height=2000)
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def transparent_rgba_path(tmp_path: Path) -> Path:
    """Creates a 4-channel RGBA transparent PNG and returns file Path."""
    p = tmp_path / "transparent_face.png"
    rgba = create_transparent_rgba_image(width=640, height=480)
    cv2.imwrite(str(p), rgba)
    return p


@pytest.fixture
def grayscale_face_path(tmp_path: Path) -> Path:
    """Creates a 1-channel Grayscale B&W face portrait and returns file Path."""
    p = tmp_path / "grayscale_face.jpg"
    gray = create_grayscale_face_image(width=640, height=480)
    cv2.imwrite(str(p), gray)
    return p


@pytest.fixture
def ultra_high_res_path(tmp_path: Path) -> Path:
    """Creates a 4000x3000 ultra-high-resolution image and returns file Path."""
    p = tmp_path / "ultra_high_res_4000x3000.jpg"
    img = create_synthetic_face_image(width=4000, height=3000, axes=(600, 850))
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def qt_pump(headless_qapp):
    """Pumps Qt event loop for processing signals/slots."""
    def _pump(iterations: int = 5, delay_sec: float = 0.01):
        for _ in range(iterations):
            headless_qapp.processEvents()
            if delay_sec > 0:
                time.sleep(delay_sec)
    return _pump


@pytest.fixture
def face_swap_engine():
    """Provides FaceSwapEngine instance or skips if not yet implemented."""
    try:
        from deepfeik.core.engine import FaceSwapEngine
        return FaceSwapEngine()
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"FaceSwapEngine not yet available: {e}")


@pytest.fixture
def video_recorder():
    """Provides AsyncVideoRecorder instance or skips if not yet implemented."""
    try:
        from deepfeik.pipeline.recorder import AsyncVideoRecorder
        return AsyncVideoRecorder()
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"AsyncVideoRecorder not yet available: {e}")


@pytest.fixture
def clipboard_service():
    """Provides ClipboardService class or skips if not yet implemented."""
    try:
        from deepfeik.gui.clipboard import ClipboardService
        return ClipboardService
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"ClipboardService not yet available: {e}")


@pytest.fixture
def main_window(headless_qapp):
    """Provides MainWindow instance or skips if not yet implemented."""
    try:
        from deepfeik.gui.main_window import MainWindow
        win = MainWindow()
        yield win
        try:
            win.close()
        except Exception:
            pass
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"MainWindow not yet available: {e}")


@pytest.fixture
def pipeline_controller_class():
    """Provides PipelineController class or skips if not yet implemented."""
    try:
        from deepfeik.pipeline.controller import PipelineController
        return PipelineController
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"PipelineController not yet available: {e}")

