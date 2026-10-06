"""
deepfeik.gui.camera_thread

QThread-based camera capture and face-swap processing worker.
Decoupled from the GUI thread via Qt signals; the main window
receives processed BGR frames and FPS updates as Qt signals.
"""

import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

from deepfeik.pipeline.video_source import WebcamSource
from deepfeik.pipeline.frame_buffer import LatestFrameBuffer
from deepfeik.core.engine import FaceSwapEngine

logger = logging.getLogger(__name__)


class CameraThread(QThread):
    """Background thread that captures webcam frames, applies the face swap,
    and emits processed frames to the main GUI thread.

    Signals:
        frame_ready(np.ndarray): Emitted for each processed BGR frame.
        fps_updated(float): Emitted ~every second with the current processing FPS.
        camera_error(str): Emitted when the webcam cannot be opened or is lost.
    """

    frame_ready = pyqtSignal(object)   # np.ndarray
    fps_updated = pyqtSignal(float)
    camera_error = pyqtSignal(str)

    def __init__(
        self,
        engine: FaceSwapEngine,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._device_index = device_index
        self._width = width
        self._height = height
        self._stop_event = threading.Event()

        # Recording state (written by GUI thread, read by camera thread)
        self._recording_lock = threading.Lock()
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Recording control (called from main/GUI thread)
    # ------------------------------------------------------------------

    def start_recording(self, output_path: str, fps: float = 30.0) -> bool:
        """Opens a VideoWriter to record swapped frames to *output_path*.

        Returns True if recording started successfully, False otherwise.
        """
        with self._recording_lock:
            if self._video_writer is not None:
                logger.warning("Recording already in progress.")
                return False

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (self._width, self._height),
            )
            if not writer.isOpened():
                logger.error(f"Could not open VideoWriter for path: {output_path}")
                writer.release()
                return False

            self._video_writer = writer
            self._recording_path = output_path
            logger.info(f"Recording started: {output_path}")
            return True

    def stop_recording(self) -> Optional[str]:
        """Finalizes and closes the current VideoWriter.

        Returns the output path of the saved file, or None if not recording.
        """
        with self._recording_lock:
            if self._video_writer is None:
                return None
            self._video_writer.release()
            path = self._recording_path
            self._video_writer = None
            self._recording_path = None
            logger.info(f"Recording saved: {path}")
            return path

    def is_recording(self) -> bool:
        """Returns True if a recording is currently active."""
        with self._recording_lock:
            return self._video_writer is not None

    # ------------------------------------------------------------------
    # QThread lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signals the run loop to exit cleanly."""
        self._stop_event.set()

    def run(self) -> None:
        """Main capture + swap loop (runs in background thread)."""
        cam = WebcamSource(
            device_index=self._device_index,
            width=self._width,
            height=self._height,
            fps=30,
        )

        if not cam.open():
            self.camera_error.emit(
                f"Cannot open webcam (device {self._device_index}). "
                "Make sure a camera is connected and not in use by another app."
            )
            return

        self._stop_event.clear()

        # FPS tracking
        fps_frame_count = 0
        fps_t0 = time.monotonic()

        try:
            while not self._stop_event.is_set():
                ok, raw_frame = cam.read()
                if not ok or raw_frame is None:
                    # Brief back-off on read failures to avoid busy loop
                    time.sleep(0.005)
                    continue

                # Mirror horizontally so it feels like a selfie camera
                raw_frame = cv2.flip(raw_frame, 1)

                # Apply face swap (returns original frame if no source loaded)
                try:
                    processed = self._engine.process_frame(raw_frame)
                except Exception as exc:
                    logger.exception(f"FaceSwapEngine error: {exc}")
                    processed = raw_frame

                # Ensure output is exactly the display size
                h, w = processed.shape[:2]
                if w != self._width or h != self._height:
                    processed = cv2.resize(processed, (self._width, self._height))

                # Write to recording if active
                with self._recording_lock:
                    if self._video_writer is not None:
                        self._video_writer.write(processed)

                # Emit frame to GUI
                self.frame_ready.emit(processed)

                # FPS update
                fps_frame_count += 1
                elapsed = time.monotonic() - fps_t0
                if elapsed >= 1.0:
                    self.fps_updated.emit(fps_frame_count / elapsed)
                    fps_frame_count = 0
                    fps_t0 = time.monotonic()

        finally:
            # Ensure recording is finalized on unexpected exit
            self.stop_recording()
            cam.release()
            self._engine.reset()
            logger.info("CameraThread stopped.")
