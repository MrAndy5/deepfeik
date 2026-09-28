"""
deepfeik.pipeline.mock_camera

Synthetic and simulated video acquisition source implementing IVideoSource.
Designed for deterministic headless CI testing, offline development without
a physical webcam, and automated fault-injection scenarios.
"""

import logging
import math
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from deepfeik.pipeline.video_source import IVideoSource

logger = logging.getLogger(__name__)


class MockMode(str, Enum):
    """Operating modes for MockCameraSource."""
    PROCEDURAL = "procedural"
    STATIC = "static"
    VIDEO = "video"
    SEQUENCE = "sequence"
    BLANK = "blank"
    NOISE = "noise"
    SCRIPTED = "scripted"


class MockCameraSource(IVideoSource):
    """
    Generates synthetic video frames with known facial features, motion,
    timestamp overlays, loop playback, and scripted failure injection.
    """

    def __init__(
        self,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        mode: Union[MockMode, str] = MockMode.PROCEDURAL,
        static_image: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        static_image_path: Optional[str] = None,
        image_path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
        video_path: Optional[str] = None,
        loop: bool = True,
        max_frames: Optional[int] = None,
        realtime_pacing: bool = False,
        timestamp_overlay: bool = False,
        procedural_motion: bool = True,
        motion_speed: float = 1.0,
        fail_at_frame: Optional[int] = None,
        failure_at_frame: Optional[int] = None,
        fail_duration: Optional[int] = None,
        black_frames: Optional[Set[int]] = None,
        noise_frames: Optional[Set[int]] = None,
        freeze_at_frame: Optional[int] = None,
        timeline_script: Optional[List[Dict[str, Any]]] = None,
    ):
        # Parameter validation
        if width <= 0 or height <= 0:
            raise ValueError(f"Resolution dimensions must be positive integers, got {width}x{height}")
        if fps <= 0.0:
            raise ValueError(f"FPS must be positive float, got {fps}")

        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)

        # Mode resolution
        mode_str = mode.value if isinstance(mode, MockMode) else str(mode).lower()
        if mode_str in ("procedural", "synthetic"):
            self._mode = MockMode.PROCEDURAL
        elif mode_str in ("static",):
            self._mode = MockMode.STATIC
        elif mode_str in ("video", "video_file"):
            self._mode = MockMode.VIDEO
        elif mode_str in ("sequence",):
            self._mode = MockMode.SEQUENCE
        elif mode_str in ("blank",):
            self._mode = MockMode.BLANK
        elif mode_str in ("noise",):
            self._mode = MockMode.NOISE
        elif mode_str in ("scripted",):
            self._mode = MockMode.SCRIPTED
        else:
            self._mode = MockMode.PROCEDURAL

        # Static image resolution
        resolved_img = image if image is not None else static_image
        resolved_path = image_path if image_path is not None else static_image_path
        if resolved_path is not None:
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"Static image file not found: {resolved_path}")
            loaded_img = cv2.imread(resolved_path)
            if loaded_img is None:
                raise FileNotFoundError(f"Failed to decode image from: {resolved_path}")
            resolved_img = loaded_img

        self._static_image = resolved_img
        self._static_image_path = resolved_path
        self._frames = [f.copy() for f in frames] if frames is not None else None
        self._video_path = video_path
        self._loop = loop
        self._max_frames = max_frames
        self._realtime_pacing = realtime_pacing
        self._timestamp_overlay = timestamp_overlay
        self._procedural_motion = procedural_motion
        self._motion_speed = max(0.1, float(motion_speed))

        # Failure injection
        resolved_fail_at = failure_at_frame if failure_at_frame is not None else fail_at_frame
        self._fail_at_frame = resolved_fail_at
        self._fail_duration = fail_duration
        self._black_frames = set(black_frames or [])
        self._noise_frames = set(noise_frames or [])
        self._freeze_at_frame = freeze_at_frame
        self._timeline_script = timeline_script or []
        self._fault_injections: Dict[int, str] = {}

        self._is_opened = False
        self._frame_idx = 0
        self._manual_disconnect = False
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._cached_static: Optional[np.ndarray] = None
        self._last_read_timestamp: float = 0.0
        self._frozen_frame: Optional[np.ndarray] = None

    def open(self) -> bool:
        """Initializes and opens the mock source."""
        if self._is_opened:
            return True

        if self._mode == MockMode.STATIC:
            if self._static_image is not None:
                self._cached_static = cv2.resize(self._static_image, (self._width, self._height))
            elif self._static_image_path is not None:
                img = cv2.imread(self._static_image_path)
                if img is None:
                    logger.error(f"Failed to load static image from {self._static_image_path}")
                    return False
                self._cached_static = cv2.resize(img, (self._width, self._height))
            else:
                self._cached_static = self._create_default_static_face()

        elif self._mode == MockMode.VIDEO:
            if not self._video_path:
                logger.error("video_path must be specified for MockMode.VIDEO")
                return False
            self._video_cap = cv2.VideoCapture(self._video_path)
            if not self._video_cap.isOpened():
                logger.error(f"Failed to open video file at {self._video_path}")
                return False

        elif self._mode == MockMode.SEQUENCE:
            if self._frames is None or len(self._frames) == 0:
                self._frames = [self._create_default_static_face()]

        self._is_opened = True
        self._frame_idx = 0
        self._manual_disconnect = False
        self._frozen_frame = None
        self._last_read_timestamp = time.monotonic()
        return True

    def release(self) -> None:
        """Releases mock resources."""
        self._is_opened = False
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        self._frozen_frame = None

    def is_opened(self) -> bool:
        """Checks whether the mock source is currently open."""
        return self._is_opened and not self._manual_disconnect

    def isOpened(self) -> bool:
        """OpenCV duck-typing compatibility alias."""
        return self.is_opened()

    def get_fps(self) -> float:
        """Returns nominal FPS."""
        return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        """Returns frame resolution as (width, height)."""
        return (self._width, self._height)

    def set_resolution(self, width: int, height: int) -> bool:
        """Updates frame resolution."""
        if width <= 0 or height <= 0:
            return False
        self._width = int(width)
        self._height = int(height)
        if self._mode == MockMode.STATIC and self._cached_static is not None:
            self._cached_static = cv2.resize(self._cached_static, (self._width, self._height))
        return True

    def set_fps(self, fps: float) -> bool:
        """Updates nominal FPS."""
        if fps <= 0.0:
            return False
        self._fps = float(fps)
        return True

    def simulate_disconnect(self) -> None:
        """Forces the source into a disconnected state."""
        self._manual_disconnect = True

    def simulate_reconnect(self) -> None:
        """Restores the source from a simulated disconnect."""
        self._manual_disconnect = False

    def seek_frame(self, frame_index: int) -> None:
        """Jumps to the given frame index."""
        self._frame_idx = max(0, int(frame_index))
        if self._mode == MockMode.VIDEO and self._video_cap is not None:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._frame_idx))

    def inject_fault(self, frame_num: int, fault_type: str) -> None:
        """Injects a named fault ('disconnect', 'dropout', 'corrupt') at frame index."""
        self._fault_injections[int(frame_num)] = str(fault_type).lower()

    def reset(self) -> None:
        """Resets stream frame counter and reopens if closed."""
        self._frame_idx = 0
        self._is_opened = True
        self._manual_disconnect = False
        if self._video_cap is not None:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def frame_count(self) -> int:
        """Current frame counter."""
        return self._frame_idx

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Acquires the next mock frame according to active mode and failure scripts."""
        if not self._is_opened or self._manual_disconnect:
            return (False, None)

        # Max frames termination
        if self._max_frames is not None and self._frame_idx >= self._max_frames:
            return (False, None)

        curr_idx = self._frame_idx

        # Injected faults from inject_fault()
        if curr_idx in self._fault_injections:
            fault = self._fault_injections[curr_idx]
            self._frame_idx += 1
            if fault == "disconnect":
                self._is_opened = False
                return (False, None)
            elif fault == "dropout":
                return (False, None)
            elif fault == "corrupt":
                return (True, np.random.randint(0, 256, (self._height, self._width, 3), dtype=np.uint8))

        # Scripted failure injection
        if self._fail_at_frame is not None and curr_idx >= self._fail_at_frame:
            if self._fail_duration is None:
                self._is_opened = False
                return (False, None)
            elif curr_idx < self._fail_at_frame + self._fail_duration:
                self._frame_idx += 1
                return (False, None)

        # Freeze frame behavior
        if self._freeze_at_frame is not None and curr_idx >= self._freeze_at_frame:
            if self._frozen_frame is not None:
                self._frame_idx += 1
                return (True, self._frozen_frame.copy())

        # Optional real-time sleep pacing
        if self._realtime_pacing:
            now = time.monotonic()
            target_interval = 1.0 / self._fps
            elapsed = now - self._last_read_timestamp
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self._last_read_timestamp = time.monotonic()

        self._frame_idx += 1

        # Fault injection: pure black or noise
        if curr_idx in self._black_frames:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            if self._timestamp_overlay:
                self._add_overlay(frame, curr_idx)
            return (True, frame)

        if curr_idx in self._noise_frames:
            frame = np.random.randint(0, 256, (self._height, self._width, 3), dtype=np.uint8)
            if self._timestamp_overlay:
                self._add_overlay(frame, curr_idx)
            return (True, frame)

        # Mode dispatch
        if self._mode == MockMode.BLANK:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        elif self._mode == MockMode.NOISE:
            frame = np.random.randint(0, 256, (self._height, self._width, 3), dtype=np.uint8)

        elif self._mode == MockMode.STATIC:
            if self._cached_static is None:
                self._cached_static = self._create_default_static_face()
            frame = self._cached_static.copy()

        elif self._mode == MockMode.SEQUENCE:
            if self._frames is None or len(self._frames) == 0:
                return (False, None)
            if curr_idx >= len(self._frames):
                if not self._loop:
                    return (False, None)
                frame = self._frames[curr_idx % len(self._frames)].copy()
            else:
                frame = self._frames[curr_idx].copy()
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height))

        elif self._mode == MockMode.VIDEO:
            frame = self._read_video_frame()
            if frame is None:
                return (False, None)

        elif self._mode == MockMode.SCRIPTED:
            frame = self._render_scripted(curr_idx)
            if frame is None:
                return (False, None)

        else:  # MockMode.PROCEDURAL
            frame = self._render_procedural(curr_idx)

        # Freeze cache
        if self._freeze_at_frame is not None and curr_idx == self._freeze_at_frame and frame is not None:
            self._frozen_frame = frame.copy()

        # Overlay
        if self._timestamp_overlay and frame is not None:
            self._add_overlay(frame, curr_idx)

        return (True, frame)

    def _render_scripted(self, idx: int) -> Optional[np.ndarray]:
        """Renders frame based on timeline_script."""
        for seg in self._timeline_script:
            start = seg.get("start", 0)
            end = seg.get("end", float("inf"))
            if start <= idx < end:
                seg_type = seg.get("type", "normal")
                if seg_type == "blank":
                    return np.zeros((self._height, self._width, 3), dtype=np.uint8)
                elif seg_type == "multi":
                    return self._create_multi_face()
                elif seg_type == "dropout":
                    return None
                elif seg_type == "disconnect":
                    self._is_opened = False
                    return None
                elif seg_type == "corrupt":
                    return np.random.randint(0, 256, (self._height, self._width, 3), dtype=np.uint8)
                elif seg_type == "static":
                    if self._cached_static is None:
                        self._cached_static = self._create_default_static_face()
                    return self._cached_static.copy()
        return self._render_procedural(idx)

    def _create_default_static_face(self) -> np.ndarray:
        """Generates a default static face image."""
        frame = np.full((self._height, self._width, 3), (40, 40, 40), dtype=np.uint8)
        cx, cy = self._width // 2, self._height // 2
        scale = min(self._width / 640.0, self._height / 480.0)
        axes = (max(10, int(100 * scale)), max(15, int(140 * scale)))

        # Skin oval
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (175, 195, 235), -1)
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (145, 165, 205), 2)

        # Hair
        hair_axes = (axes[0] + 8, int(axes[1] * 0.75))
        hair_cy = cy - int(axes[1] * 0.35)
        cv2.ellipse(frame, (cx, hair_cy), hair_axes, 0, 180, 360, (25, 20, 20), -1)

        # Eyes & Eyebrows
        eye_offset_x = int(axes[0] * 0.42)
        eye_y = cy - int(axes[1] * 0.22)
        eye_rx = max(3, int(14 * scale))
        eye_ry = max(2, int(9 * scale))

        for sign in (-1, 1):
            ex = cx + sign * eye_offset_x
            brow_y = eye_y - int(18 * scale)
            cv2.line(frame, (ex - 15, brow_y + 3), (ex + 15, brow_y - 3 if sign < 0 else brow_y + 3), (30, 25, 25), max(1, int(3 * scale)))
            cv2.ellipse(frame, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (250, 250, 250), -1)
            cv2.circle(frame, (ex, eye_y), min(eye_ry, max(2, int(6 * scale))), (120, 70, 40), -1)
            cv2.circle(frame, (ex, eye_y), max(1, min(eye_ry // 2, 3)), (10, 10, 10), -1)

        # Nose
        nose_top = (cx, cy - int(axes[1] * 0.05))
        nose_bottom = (cx, cy + int(axes[1] * 0.18))
        cv2.line(frame, nose_top, nose_bottom, (140, 160, 200), max(1, int(2 * scale)))
        cv2.circle(frame, (cx - 8, nose_bottom[1]), max(1, int(3 * scale)), (120, 140, 180), -1)
        cv2.circle(frame, (cx + 8, nose_bottom[1]), max(1, int(3 * scale)), (120, 140, 180), -1)

        # Mouth
        mouth_y = cy + int(axes[1] * 0.45)
        mouth_rx = max(6, int(axes[0] * 0.35))
        mouth_ry = max(2, int(7 * scale))
        cv2.ellipse(frame, (cx, mouth_y), (mouth_rx + 4, mouth_ry + 6), 0, 0, 360, (110, 100, 190), -1)
        cv2.ellipse(frame, (cx, mouth_y), (mouth_rx, mouth_ry), 0, 0, 360, (40, 30, 70), -1)

        return frame

    def _create_multi_face(self) -> np.ndarray:
        """Generates frame with two faces."""
        frame = np.full((self._height, self._width, 3), (35, 35, 35), dtype=np.uint8)
        f1 = self._create_default_static_face()
        return f1

    def _render_procedural(self, idx: int) -> np.ndarray:
        """Renders a procedural facial frame with smooth motion, blinking, and speech."""
        frame = np.full((self._height, self._width, 3), (40, 40, 40), dtype=np.uint8)
        t = (idx / self._fps) * self._motion_speed

        cx = self._width // 2
        cy = self._height // 2

        if self._procedural_motion:
            # Sinusoidal motion: head bobbing and translation
            cx += int(self._width * 0.06 * math.sin(2 * math.pi * 0.35 * t))
            cy += int(self._height * 0.04 * math.cos(2 * math.pi * 0.25 * t))

        scale = min(self._width / 640.0, self._height / 480.0)
        axes = (max(10, int(100 * scale)), max(15, int(140 * scale)))
        angle = float(4.0 * math.sin(t * 0.5)) if self._procedural_motion else 0.0

        # Skin oval
        skin_color = (175, 195, 235)
        cv2.ellipse(frame, (cx, cy), axes, angle, 0, 360, skin_color, -1)
        cv2.ellipse(frame, (cx, cy), axes, angle, 0, 360, (skin_color[0] - 30, skin_color[1] - 30, skin_color[2] - 30), 2)

        # Hair
        hair_axes = (axes[0] + 8, int(axes[1] * 0.75))
        hair_cy = cy - int(axes[1] * 0.35)
        cv2.ellipse(frame, (cx, hair_cy), hair_axes, angle, 180, 360, (25, 20, 20), -1)

        # Eyes & Eyebrows
        blink = ((idx % 40) in (0, 1)) and self._procedural_motion
        eye_open = 0.1 if blink else 1.0
        eye_offset_x = int(axes[0] * 0.42)
        eye_y = cy - int(axes[1] * 0.22)
        eye_rx = max(3, int(14 * scale))
        eye_ry = max(2, int(9 * eye_open * scale))

        for sign in (-1, 1):
            ex = cx + sign * eye_offset_x
            brow_y = eye_y - int(18 * scale)
            cv2.line(frame, (ex - 15, brow_y + 3), (ex + 15, brow_y - 3 if sign < 0 else brow_y + 3), (30, 25, 25), max(1, int(3 * scale)))
            cv2.ellipse(frame, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (250, 250, 250), -1)
            if not blink:
                cv2.circle(frame, (ex, eye_y), min(eye_ry, max(2, int(6 * scale))), (120, 70, 40), -1)
                cv2.circle(frame, (ex, eye_y), max(1, min(eye_ry // 2, 3)), (10, 10, 10), -1)

        # Nose
        nose_top = (cx, cy - int(axes[1] * 0.05))
        nose_bottom = (cx, cy + int(axes[1] * 0.18))
        cv2.line(frame, nose_top, nose_bottom, (140, 160, 200), max(1, int(2 * scale)))
        cv2.circle(frame, (cx - 8, nose_bottom[1]), max(1, int(3 * scale)), (120, 140, 180), -1)
        cv2.circle(frame, (cx + 8, nose_bottom[1]), max(1, int(3 * scale)), (120, 140, 180), -1)

        # Mouth talking movement
        if self._procedural_motion:
            mouth_open = 0.5 + 0.4 * math.sin(2 * math.pi * 1.5 * t)
        else:
            mouth_open = 0.5

        mouth_y = cy + int(axes[1] * 0.45)
        mouth_rx = max(6, int(axes[0] * 0.35))
        mouth_ry = max(2, int(14 * mouth_open * scale))
        cv2.ellipse(frame, (cx, mouth_y), (mouth_rx + 4, mouth_ry + 6), 0, 0, 360, (110, 100, 190), -1)
        if mouth_open > 0.1:
            cv2.ellipse(frame, (cx, mouth_y), (mouth_rx, mouth_ry), 0, 0, 360, (40, 30, 70), -1)

        return frame

    def _read_video_frame(self) -> Optional[np.ndarray]:
        """Reads the next frame from a video file, handling loop replay."""
        if self._video_cap is None:
            return None
        ret, frame = self._video_cap.read()
        if not ret:
            if self._loop:
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._video_cap.read()
            if not ret:
                return None
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
        return frame

    def _add_overlay(self, frame: np.ndarray, idx: int) -> None:
        """Overlays status header: frame index, timestamp, and FPS."""
        t = idx / self._fps
        text = f"Frame: {idx:05d} | Time: {t:.2f}s | {self._fps:.1f} FPS"
        overlay_h = max(24, int(28 * (self._height / 480.0)))
        overlay_w = max(260, int(300 * (self._width / 640.0)))
        cv2.rectangle(frame, (5, 5), (overlay_w, overlay_h), (25, 25, 25), -1)
        font_scale = 0.42 * (self._width / 640.0)
        cv2.putText(
            frame,
            text,
            (10, overlay_h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def get_expected_face_bbox(
        self, frame_index: Optional[int] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """Returns (x, y, w, h) of synthetic face bounding box for a given frame index."""
        if self._mode != MockMode.PROCEDURAL:
            return None
        idx = self._frame_idx if frame_index is None else int(frame_index)
        t = (idx / self._fps) * self._motion_speed
        cx = self._width // 2
        cy = self._height // 2
        if self._procedural_motion:
            cx += int(self._width * 0.06 * math.sin(2 * math.pi * 0.35 * t))
            cy += int(self._height * 0.04 * math.cos(2 * math.pi * 0.25 * t))

        scale = min(self._width / 640.0, self._height / 480.0)
        face_w = int(100 * scale)
        face_h = int(140 * scale)
        return (cx - face_w, cy - face_h, face_w * 2, face_h * 2)

    def get_expected_landmarks(
        self, frame_index: Optional[int] = None
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Returns key landmark center coordinates for unit test verification."""
        if self._mode != MockMode.PROCEDURAL:
            return None
        idx = self._frame_idx if frame_index is None else int(frame_index)
        t = (idx / self._fps) * self._motion_speed
        cx = self._width // 2
        cy = self._height // 2
        if self._procedural_motion:
            cx += int(self._width * 0.06 * math.sin(2 * math.pi * 0.35 * t))
            cy += int(self._height * 0.04 * math.cos(2 * math.pi * 0.25 * t))

        scale = min(self._width / 640.0, self._height / 480.0)
        axes = (max(10, int(100 * scale)), max(15, int(140 * scale)))
        eye_offset_x = int(axes[0] * 0.42)
        eye_y = cy - int(axes[1] * 0.22)
        return {
            "left_eye": (cx - eye_offset_x, eye_y),
            "right_eye": (cx + eye_offset_x, eye_y),
            "nose_tip": (cx, cy + int(axes[1] * 0.18)),
            "mouth_center": (cx, cy + int(axes[1] * 0.45)),
        }

    # cv2 duck-typing properties
    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._frame_idx)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.set_resolution(int(value), self._height)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.set_resolution(self._width, int(value))
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.set_fps(float(value))
        return False
