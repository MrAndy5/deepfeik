"""
deepfeik.pipeline.video_source

Defines the abstract video source interface (IVideoSource) and the primary
hardware webcam implementation (WebcamSource) with Windows DirectShow
optimization, automatic fallback, resolution configuration, and disconnect detection.
"""

import logging
import platform
import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class IVideoSource(ABC):
    """
    Abstract base interface for all video acquisition sources.

    Decouples video capture devices and synthetic generators from downstream
    face processing, live GUI preview, and video recording.
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Initializes and opens the video source.

        Returns:
            bool: True if opened successfully and ready to stream, False otherwise.
        """
        ...

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Acquires the next video frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - (True, frame): Frame successfully read. `frame` is a BGR uint8
                  NumPy array of shape (height, width, 3).
                - (False, None): Capture failed, stream reached EOF, or device was disconnected.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """
        Releases underlying capture resources and hardware handles.
        Must be idempotent (safe to call multiple times without error).
        """
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Checks whether the video source is currently opened and available.

        Returns:
            bool: True if source is initialized and streaming, False otherwise.
        """
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """
        Returns the nominal or configured frames-per-second of the source.

        Returns:
            float: Nominal frame rate (e.g., 30.0). Always returns a positive
            non-zero float, falling back to a sensible default if the underlying
            driver reports -1.0 or 0.0.
        """
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Returns the video frame resolution as (width, height).

        Returns:
            Tuple[int, int]: (width, height) in pixels.
        """
        ...

    def isOpened(self) -> bool:
        """OpenCV duck-typing compatibility alias for is_opened()."""
        return self.is_opened()

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Configures the frame resolution.

        Args:
            width: Desired frame width in pixels.
            height: Desired frame height in pixels.

        Returns:
            bool: True if configuration was applied, False otherwise.
        """
        return False

    def set_fps(self, fps: float) -> bool:
        """
        Configures the capture frame rate.

        Args:
            fps: Desired frame rate in frames per second.

        Returns:
            bool: True if configuration was applied, False otherwise.
        """
        return False

    def __enter__(self) -> "IVideoSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()


class WebcamSource(IVideoSource):
    """
    Hardware webcam acquisition source using OpenCV VideoCapture.

    Features:
    - Windows DirectShow (cv2.CAP_DSHOW) optimization for fast startup (<0.7s)
      and stable frame acquisition.
    - Automatic fallback to cv2.CAP_ANY if DirectShow fails or on non-Windows OS.
    - Resolution and FPS property application with actual hardware query.
    - Robust handling of DirectShow -1.0 FPS reporting.
    - Disconnect detection after configurable consecutive read failures.
    - Graceful reconnection capability.
    - Idempotent resource release and destructor protection.
    """

    def __init__(
        self,
        device_index: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        backend: Optional[int] = None,
        auto_fallback: bool = True,
        disconnect_threshold: int = 5,
    ):
        """
        Initializes WebcamSource configuration.

        Args:
            device_index: Integer index of webcam device (0 = default), or video
              device path / stream URL string.
            width: Requested horizontal frame resolution.
            height: Requested vertical frame resolution.
            fps: Requested capture frame rate.
            backend: Explicit OpenCV VideoCapture backend flag (e.g., cv2.CAP_DSHOW,
              cv2.CAP_MSMF, cv2.CAP_ANY). If None, defaults to CAP_DSHOW on Windows
              and CAP_ANY on other platforms.
            auto_fallback: If True and the selected backend fails to open, automatically
              retries opening with cv2.CAP_ANY.
            disconnect_threshold: Number of consecutive failed reads before the source
              is flagged as disconnected.
        """
        self._device_index = device_index
        self._requested_width = max(1, int(width))
        self._requested_height = max(1, int(height))
        self._requested_fps = max(1.0, float(fps))
        self._backend = backend
        self._auto_fallback = auto_fallback
        self._disconnect_threshold = max(1, int(disconnect_threshold))

        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_width: int = self._requested_width
        self._actual_height: int = self._requested_height
        self._actual_fps: float = self._requested_fps
        self._consecutive_failures: int = 0
        self._is_connected: bool = False

    def open(self) -> bool:
        """
        Opens the webcam device handle with backend selection and fallback.

        Returns:
            bool: True if opened successfully, False otherwise.
        """
        if self.is_opened():
            return True

        selected_backend = self._backend
        if selected_backend is None:
            if platform.system() == "Windows":
                selected_backend = cv2.CAP_DSHOW
            else:
                selected_backend = cv2.CAP_ANY

        try:
            cap = cv2.VideoCapture(self._device_index, selected_backend)
            if not cap.isOpened() and self._auto_fallback and selected_backend != cv2.CAP_ANY:
                logger.warning(
                    f"Failed to open webcam {self._device_index} with backend {selected_backend}; "
                    f"retrying with CAP_ANY."
                )
                cap.release()
                cap = cv2.VideoCapture(self._device_index, cv2.CAP_ANY)

            if not cap.isOpened():
                logger.error(f"Failed to open webcam with device_index={self._device_index}.")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._cap = None
                self._is_connected = False
                return False

            self._cap = cap
            self._apply_properties()
            self._consecutive_failures = 0
            self._is_connected = True
            logger.info(
                f"Webcam {self._device_index} opened: {self._actual_width}x{self._actual_height} "
                f"@ {self._actual_fps:.1f} FPS"
            )
            return True

        except Exception as exc:
            logger.exception(f"Unexpected exception opening webcam {self._device_index}: {exc}")
            self.release()
            return False

    def _apply_properties(self) -> None:
        """Configures hardware properties on the underlying VideoCapture."""
        if self._cap is None:
            return

        # Set requested resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        # Read back actual resolution clamped by hardware driver
        try:
            act_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            act_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if act_w > 0 and act_h > 0:
                self._actual_width = act_w
                self._actual_height = act_h
            else:
                self._actual_width = self._requested_width
                self._actual_height = self._requested_height
        except (TypeError, ValueError):
            self._actual_width = self._requested_width
            self._actual_height = self._requested_height

        # Read back FPS, handling Windows DirectShow -1.0 or 0.0
        try:
            hw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            if hw_fps > 0.0 and not np.isnan(hw_fps) and not np.isinf(hw_fps):
                self._actual_fps = hw_fps
            else:
                self._actual_fps = self._requested_fps
        except (TypeError, ValueError):
            self._actual_fps = self._requested_fps

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single frame from the webcam.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, bgr_frame) or (False, None).
        """
        if not self.is_opened() or self._cap is None:
            return (False, None)

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._disconnect_threshold:
                    if self._is_connected:
                        self._is_connected = False
                        logger.warning(
                            f"Webcam {self._device_index} disconnected after "
                            f"{self._consecutive_failures} consecutive read failures."
                        )
                return (False, None)

            self._consecutive_failures = 0
            self._is_connected = True
            return (True, frame)

        except Exception as exc:
            self._consecutive_failures += 1
            logger.error(f"Exception during webcam read: {exc}")
            return (False, None)

    def release(self) -> None:
        """Closes and releases the underlying VideoCapture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.debug(f"Error releasing VideoCapture handle: {exc}")
            self._cap = None
        self._is_connected = False
        self._consecutive_failures = 0

    def is_opened(self) -> bool:
        """Checks whether the VideoCapture handle is valid and opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_connected(self) -> bool:
        """Returns True if the webcam is open and not stalled by read failures."""
        return self.is_opened() and self._is_connected

    def get_fps(self) -> float:
        """Returns the effective capture frame rate."""
        if self.is_opened():
            return self._actual_fps
        return self._requested_fps

    def get_resolution(self) -> Tuple[int, int]:
        """Returns the effective frame resolution as (width, height)."""
        if self.is_opened():
            return (self._actual_width, self._actual_height)
        return (self._requested_width, self._requested_height)

    def set_resolution(self, width: int, height: int) -> bool:
        """Reconfigures the capture resolution."""
        self._requested_width = max(1, int(width))
        self._requested_height = max(1, int(height))
        if self.is_opened():
            self._apply_properties()
        return True

    def set_fps(self, fps: float) -> bool:
        """Reconfigures the capture frame rate."""
        self._requested_fps = max(1.0, float(fps))
        if self.is_opened():
            self._apply_properties()
        return True

    def reconnect(self) -> bool:
        """Closes the current capture and attempts to re-open."""
        self.release()
        time.sleep(0.05)
        return self.open()

    def __del__(self) -> None:
        self.release()
