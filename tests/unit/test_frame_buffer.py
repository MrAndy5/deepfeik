"""Unit tests for LatestFrameBuffer (single-slot drop-oldest buffer)."""
import time
import threading
import pytest
import numpy as np

from deepfeik.pipeline.frame_buffer import LatestFrameBuffer


# ============================================================================
# 1. Single-Slot State and Overwrite Tests
# ============================================================================

def test_frame_buffer_initial_state():
    """New LatestFrameBuffer is empty and dropped_count is 0."""
    buf = LatestFrameBuffer()
    assert buf.is_empty() is True
    assert buf.dropped_count == 0
    assert buf.get(timeout=0.01) is None


def test_frame_buffer_put_and_get_single_frame(sample_frame):
    """Pushed frame can be retrieved, leaving buffer empty."""
    buf = LatestFrameBuffer()
    buf.put(sample_frame)

    assert buf.is_empty() is False
    assert buf.dropped_count == 0

    frame = buf.get(timeout=0.1)
    assert frame is not None
    assert np.array_equal(frame, sample_frame)
    assert buf.is_empty() is True


def test_frame_buffer_single_slot_overwrite_drop_oldest():
    """Overwriting an unread frame discards it and increments dropped_count."""
    buf = LatestFrameBuffer()
    f1 = np.full((10, 10, 3), 1, dtype=np.uint8)
    f2 = np.full((10, 10, 3), 2, dtype=np.uint8)
    f3 = np.full((10, 10, 3), 3, dtype=np.uint8)

    buf.put(f1)
    assert buf.dropped_count == 0

    buf.put(f2)
    assert buf.dropped_count == 1

    buf.put(f3)
    assert buf.dropped_count == 2

    # Consumer retrieves the latest frame (f3)
    retrieved = buf.get(timeout=0.1)
    assert np.array_equal(retrieved, f3)
    assert buf.is_empty() is True
    assert buf.get(timeout=0.01) is None


def test_frame_buffer_timeout_on_empty():
    """get() times out gracefully when buffer remains empty."""
    buf = LatestFrameBuffer()
    t0 = time.perf_counter()
    result = buf.get(timeout=0.05)
    elapsed = time.perf_counter() - t0

    assert result is None
    assert 0.035 <= elapsed <= 0.20, f"Timeout took {elapsed:.4f}s, expected ~0.05s"


def test_frame_buffer_clear():
    """clear() discards any buffered frame and unsets ready flag."""
    buf = LatestFrameBuffer()
    buf.put(np.zeros((10, 10, 3), dtype=np.uint8))
    assert buf.is_empty() is False

    buf.clear()
    assert buf.is_empty() is True
    assert buf.get(timeout=0.01) is None


def test_frame_buffer_put_none_rejected():
    """put(None) raises ValueError to guard against invalid frame data."""
    buf = LatestFrameBuffer()
    with pytest.raises(ValueError):
        buf.put(None)


def test_frame_buffer_reset_drop_count():
    """dropped_count can be queried and optionally reset."""
    buf = LatestFrameBuffer()
    buf.put(np.zeros((5, 5, 3), dtype=np.uint8))
    buf.put(np.zeros((5, 5, 3), dtype=np.uint8))
    assert buf.dropped_count == 1

    if hasattr(buf, "reset_dropped_count"):
        buf.reset_dropped_count()
        assert buf.dropped_count == 0


# ============================================================================
# 2. Concurrency & Multi-Threading Safety Tests
# ============================================================================

def test_frame_buffer_producer_consumer_monotone_ordering():
    """High-speed producer and consumer maintain strictly increasing sequence IDs."""
    buf = LatestFrameBuffer()
    total_frames = 150
    consumed_ids = []
    stop_event = threading.Event()

    def producer():
        for i in range(total_frames):
            frame = np.full((10, 10, 3), i, dtype=np.uint8)
            buf.put(frame)
            time.sleep(0.001)  # ~1000 FPS burst
        stop_event.set()

    def consumer():
        while not stop_event.is_set() or not buf.is_empty():
            frame = buf.get(timeout=0.01)
            if frame is not None:
                consumed_ids.append(int(frame[0, 0, 0]))
                time.sleep(0.003)  # Slower consumer (~300 FPS)

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)
    t_prod.start()
    t_cons.start()
    t_prod.join(timeout=3.0)
    t_cons.join(timeout=3.0)

    assert len(consumed_ids) > 0
    # Assert strict monotonicity: consumer never sees older frames out of order
    for i in range(1, len(consumed_ids)):
        assert consumed_ids[i] > consumed_ids[i - 1], "Frames arrived out of order"

    # Frame conservation law: consumed + dropped == total
    total_accounted = len(consumed_ids) + buf.dropped_count
    assert total_accounted == total_frames, (
        f"Frame accounting mismatch: {len(consumed_ids)} consumed + {buf.dropped_count} dropped "
        f"!= {total_frames} produced"
    )


def test_frame_buffer_zero_latency_guarantee():
    """Fast burst of frames guarantees consumer immediately receives the latest frame."""
    buf = LatestFrameBuffer()
    # Producer pushes 50 frames rapidly
    for i in range(50):
        buf.put(np.full((10, 10, 3), i, dtype=np.uint8))

    latest = buf.get(timeout=0.05)
    assert latest is not None
    assert latest[0, 0, 0] == 49, "Buffer must yield the most recent frame"
    assert buf.dropped_count == 49


def test_frame_buffer_consumer_faster_than_producer():
    """When consumer is faster than producer, 0 frames are dropped."""
    buf = LatestFrameBuffer()
    frames_to_send = 10
    received = []

    def slow_producer():
        for i in range(frames_to_send):
            time.sleep(0.01)
            buf.put(np.full((5, 5, 3), i, dtype=np.uint8))

    t_prod = threading.Thread(target=slow_producer)
    t_prod.start()

    for _ in range(frames_to_send):
        f = buf.get(timeout=0.2)
        assert f is not None
        received.append(int(f[0, 0, 0]))

    t_prod.join(timeout=1.0)
    assert received == list(range(frames_to_send))
    assert buf.dropped_count == 0


def test_frame_buffer_concurrent_producers():
    """Multiple concurrent producer threads do not deadlock or corrupt state."""
    buf = LatestFrameBuffer()
    num_threads = 4
    frames_per_thread = 25

    def worker(tid):
        for i in range(frames_per_thread):
            buf.put(np.full((5, 5, 3), tid * 25 + i, dtype=np.uint8))
            time.sleep(0.0005)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    # Buffer should hold exactly 1 valid frame and correctly report dropped count
    assert buf.is_empty() is False
    total_produced = num_threads * frames_per_thread
    assert buf.dropped_count == total_produced - 1
    last_frame = buf.get(timeout=0.01)
    assert last_frame is not None
    assert buf.is_empty() is True


def test_frame_buffer_close_unblocks_waiting_get():
    """Calling close() immediately unblocks any thread waiting on get()."""
    buf = LatestFrameBuffer()
    unblocked = threading.Event()

    def waiting_consumer():
        t0 = time.perf_counter()
        _ = buf.get(timeout=5.0)  # Long timeout
        if time.perf_counter() - t0 < 1.0:
            unblocked.set()

    t = threading.Thread(target=waiting_consumer)
    t.start()
    time.sleep(0.05)  # Allow thread to enter wait
    buf.close()       # Must unblock consumer
    t.join(timeout=1.0)

    assert unblocked.is_set(), "buf.close() failed to immediately unblock waiting consumer"


def test_frame_buffer_get_nowait_and_peek(sample_frame):
    """LatestFrameBuffer get_nowait and peek methods."""
    buf = LatestFrameBuffer()
    assert buf.get_nowait() is None
    assert buf.peek() is None

    buf.put(sample_frame)
    # peek does not consume the frame
    peeked = buf.peek()
    assert peeked is not None
    assert np.array_equal(peeked, sample_frame)
    assert buf.is_empty() is False

    # get_nowait consumes the frame
    consumed = buf.get_nowait()
    assert consumed is not None
    assert np.array_equal(consumed, sample_frame)
    assert buf.is_empty() is True


def test_frame_buffer_stats_and_sequence_number(sample_frame):
    """LatestFrameBuffer stats snapshot and sequence number tracking."""
    buf = LatestFrameBuffer()
    assert buf.sequence_number == 0

    buf.put(sample_frame)
    buf.put(sample_frame)
    assert buf.sequence_number == 2

    buf.get()
    stats = buf.stats
    assert stats.produced_count == 2
    assert stats.consumed_count == 1
    assert stats.dropped_count == 1
    assert stats.drop_ratio == 0.5
    assert stats.sequence_number == 2


def test_frame_buffer_copy_on_read(sample_frame):
    """LatestFrameBuffer copy_on_read=True isolates consumer modifications."""
    buf = LatestFrameBuffer(copy_on_read=True)
    buf.put(sample_frame)

    frame1 = buf.peek()
    frame1[0, 0, 0] = 99  # modify peeked frame
    frame2 = buf.get()
    assert frame2[0, 0, 0] == sample_frame[0, 0, 0]  # original unaffected


def test_frame_buffer_context_manager_and_is_closed(sample_frame):
    """LatestFrameBuffer context manager closes cleanly."""
    with LatestFrameBuffer() as buf:
        assert buf.is_closed() is False
        buf.put(sample_frame)

    assert buf.is_closed() is True
    assert buf.put(sample_frame) is False
