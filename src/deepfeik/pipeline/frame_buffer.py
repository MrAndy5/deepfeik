"""
deepfeik.pipeline.frame_buffer

Thread-safe single-slot drop-oldest buffer (LatestFrameBuffer).
Decouples high-rate or variable-rate capture threads from downstream
deepfake inference workers, guaranteeing strictly zero frame queue latency.
"""

import logging
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferStats:
    """Diagnostic statistics for LatestFrameBuffer."""
    produced_count: int
    consumed_count: int
    dropped_count: int
    drop_ratio: float
    sequence_number: int


class BufferClosedError(RuntimeError):
    """Raised when attempting an operation on a closed buffer (if strict mode enabled)."""
    pass


class LatestFrameBuffer:
    """
    Thread-safe single-slot drop-oldest buffer guaranteeing zero frame latency.

    Key Guarantees:
    - Drop-oldest semantics: Holding only the single most recent frame. If a new frame
      arrives before the previous frame is consumed, the previous frame is discarded.
    - Zero queue latency: The consumer never experiences accumulated buffer lag.
    - Condition variable synchronization: Consumer threads wait without busy polling.
    - Clean unblocking: Calling close() immediately wakes all waiting threads with None.
    - Optional memory copying (copy_on_read) to isolate consumer mutations.
    - Thread-safe statistics tracking (produced, consumed, dropped, ratio).
    """

    def __init__(self, copy_on_read: bool = False):
        """
        Initializes LatestFrameBuffer.

        Args:
            copy_on_read: If True, get() returns an independent copy of the frame array
              (frame.copy()), preventing in-place consumer modifications from mutating
              producer memory. Defaults to False for maximum CPU throughput (zero-copy).
        """
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._slot: Optional[np.ndarray] = None
        self._has_new: bool = False
        self._closed: bool = False
        self._copy_on_read: bool = bool(copy_on_read)

        self._produced_count: int = 0
        self._consumed_count: int = 0
        self._dropped_count: int = 0
        self._sequence_number: int = 0

    def put(self, frame: np.ndarray) -> bool:
        """
        Puts a new frame into the single-slot buffer, overwriting any unread frame.

        Args:
            frame: BGR NumPy array of the video frame.

        Returns:
            bool: True if accepted, False if buffer is closed.

        Raises:
            ValueError: If frame is None.
        """
        if frame is None:
            raise ValueError("Cannot put None into LatestFrameBuffer.")

        with self._cond:
            if self._closed:
                return False

            if self._has_new:
                self._dropped_count += 1

            self._slot = frame
            self._has_new = True
            self._produced_count += 1
            self._sequence_number += 1
            self._cond.notify_all()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Retrieves and consumes the latest frame from the buffer.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely until
              a frame arrives or buffer is closed. 0 means non-blocking check.

        Returns:
            Optional[np.ndarray]: The latest video frame array, or None if timed out,
            buffer is empty (at timeout=0), or buffer is closed.
        """
        with self._cond:
            if self._closed:
                return None

            deadline = None if timeout is None else (time.monotonic() + timeout)

            while not self._has_new and not self._closed:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None

                self._cond.wait(timeout=remaining)

            if self._closed or not self._has_new:
                return None

            frame = self._slot
            self._slot = None
            self._has_new = False
            self._consumed_count += 1
            return frame.copy() if self._copy_on_read else frame

    def get_nowait(self) -> Optional[np.ndarray]:
        """
        Non-blocking retrieval of the latest frame.
        Equivalent to get(timeout=0).
        """
        return self.get(timeout=0)

    def peek(self) -> Optional[np.ndarray]:
        """
        Inspects the current frame without consuming it or clearing the new-frame flag.

        Returns:
            Optional[np.ndarray]: Current frame array, or None if empty or closed.
        """
        with self._lock:
            if not self._has_new or self._closed or self._slot is None:
                return None
            return self._slot.copy() if self._copy_on_read else self._slot

    def clear(self) -> None:
        """Discards any currently stored frame without closing the buffer."""
        with self._lock:
            if self._has_new:
                self._dropped_count += 1
            self._slot = None
            self._has_new = False

    def close(self) -> None:
        """Closes the buffer and immediately notifies all waiting threads to wake and exit."""
        with self._cond:
            if not self._closed:
                if self._has_new:
                    self._dropped_count += 1
                    self._has_new = False
                self._slot = None
                self._closed = True
                self._cond.notify_all()

    def is_closed(self) -> bool:
        """Returns True if the buffer has been closed."""
        with self._lock:
            return self._closed

    def is_empty(self) -> bool:
        """Returns True if there is currently no unconsumed frame in the buffer."""
        with self._lock:
            return not self._has_new

    @property
    def dropped_count(self) -> int:
        """Number of frames dropped due to single-slot overwrite."""
        with self._lock:
            return self._dropped_count

    def reset_dropped_count(self) -> None:
        """Resets dropped frame counter to 0."""
        with self._lock:
            self._dropped_count = 0

    @property
    def sequence_number(self) -> int:
        """Monotonically increasing sequence number of frames put into the buffer."""
        with self._lock:
            return self._sequence_number

    @property
    def stats(self) -> BufferStats:
        """Snapshot of buffer performance metrics."""
        with self._lock:
            ratio = (
                (self._dropped_count / self._produced_count)
                if self._produced_count > 0
                else 0.0
            )
            return BufferStats(
                produced_count=self._produced_count,
                consumed_count=self._consumed_count,
                dropped_count=self._dropped_count,
                drop_ratio=ratio,
                sequence_number=self._sequence_number,
            )

    def __enter__(self) -> "LatestFrameBuffer":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
