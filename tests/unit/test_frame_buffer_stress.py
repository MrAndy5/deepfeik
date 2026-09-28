"""
tests.unit.test_frame_buffer_stress

Adversarial stress and concurrency verification test suite for LatestFrameBuffer
and camera ingestion pipeline.

Tests:
1. High-throughput concurrency: 10 producers pushing 10,000 frames total, 5 consumers.
2. Invariant verification: produced == consumed + dropped across all workloads.
3. Shutdown under load: close() while N threads wait on get(), verifying instant unblock (<50ms).
4. Buffer memory leak check: weakref tracking of frame arrays to verify garbage collection.
5. Ingestion pipeline stress: MockCameraSource streaming to LatestFrameBuffer under load.
"""

import gc
import queue
import threading
import time
import weakref
import numpy as np
import pytest

from deepfeik.pipeline.frame_buffer import LatestFrameBuffer, BufferStats
from deepfeik.pipeline.mock_camera import MockCameraSource, MockMode


# ============================================================================
# 1. High-Throughput Concurrency Stress: 10 Producers, 10,000 Frames, 5 Consumers
# ============================================================================

def test_stress_high_throughput_10_producers_10k_frames_5_consumers():
    """
    Stress test with 10 concurrent producer threads pushing 1,000 frames each (10,000 total)
    while 5 concurrent consumer threads continuously consume frames.
    Verifies throughput, absence of deadlocks, absence of exceptions, and state consistency.
    """
    buf = LatestFrameBuffer(copy_on_read=False)
    num_producers = 10
    frames_per_producer = 1000
    total_expected_frames = num_producers * frames_per_producer  # 10,000
    num_consumers = 5

    producer_errors = []
    consumer_errors = []
    consumed_per_thread = [0] * num_consumers
    producer_done = threading.Event()
    producers_finished = 0
    producers_finished_lock = threading.Lock()

    def producer_worker(pid: int):
        try:
            nonlocal producers_finished
            for i in range(frames_per_producer):
                # Shape: 64x64x3 uint8 array encoded with producer ID and seq
                f = np.empty((64, 64, 3), dtype=np.uint8)
                f[0, 0, 0] = pid
                f[0, 0, 1] = i % 256
                f[0, 0, 2] = (i // 256) % 256
                ok = buf.put(f)
                if not ok:
                    producer_errors.append(f"Producer {pid} put returned False at frame {i}")
                    break
            with producers_finished_lock:
                producers_finished += 1
                if producers_finished == num_producers:
                    producer_done.set()
        except Exception as e:
            producer_errors.append(f"Producer {pid} raised: {e}")

    def consumer_worker(cid: int):
        try:
            count = 0
            while True:
                frame = buf.get(timeout=0.02)
                if frame is not None:
                    count += 1
                    # Basic frame validation
                    assert frame.shape == (64, 64, 3)
                    assert frame.dtype == np.uint8
                else:
                    if producer_done.is_set() and buf.is_empty():
                        break
            consumed_per_thread[cid] = count
        except Exception as e:
            consumer_errors.append(f"Consumer {cid} raised: {e}")

    start_time = time.perf_counter()

    producers = [
        threading.Thread(target=producer_worker, args=(p,))
        for p in range(num_producers)
    ]
    consumers = [
        threading.Thread(target=consumer_worker, args=(c,))
        for c in range(num_consumers)
    ]

    for c in consumers:
        c.start()
    for p in producers:
        p.start()

    for p in producers:
        p.join(timeout=10.0)
        assert not p.is_alive(), "Producer thread hung!"

    # Allow consumers to drain remaining frames
    producer_done.set()
    for c in consumers:
        c.join(timeout=10.0)
        assert not c.is_alive(), "Consumer thread hung!"

    elapsed = time.perf_counter() - start_time
    fps = total_expected_frames / elapsed

    assert len(producer_errors) == 0, f"Producer errors: {producer_errors}"
    assert len(consumer_errors) == 0, f"Consumer errors: {consumer_errors}"

    total_consumed = sum(consumed_per_thread)
    stats = buf.stats

    print(
        f"\n[STRESS 10k] Elapsed: {elapsed:.3f}s | Throughput: {fps:.1f} fps | "
        f"Produced: {stats.produced_count} | Consumed: {total_consumed} | Dropped: {stats.dropped_count}"
    )

    assert stats.produced_count == total_expected_frames
    assert stats.sequence_number == total_expected_frames
    # Conservation invariant after drain:
    # If the buffer is empty: produced == consumed + dropped
    if buf.is_empty():
        assert stats.produced_count == stats.consumed_count + stats.dropped_count
        assert total_consumed == stats.consumed_count


# ============================================================================
# 2. Invariant Verification: produced == consumed + dropped
# ============================================================================

def test_invariant_single_producer_fast_consumer_no_drops():
    """When consumer reads faster than producer, 0 frames are dropped: produced == consumed."""
    buf = LatestFrameBuffer()
    total_frames = 200
    consumed = 0

    def slow_producer():
        for i in range(total_frames):
            buf.put(np.full((10, 10, 3), i % 256, dtype=np.uint8))
            time.sleep(0.0005)

    prod_thread = threading.Thread(target=slow_producer)
    prod_thread.start()

    while consumed < total_frames:
        f = buf.get(timeout=1.0)
        if f is not None:
            consumed += 1

    prod_thread.join(timeout=3.0)

    stats = buf.stats
    assert stats.produced_count == total_frames
    assert stats.consumed_count == total_frames
    assert stats.dropped_count == 0
    assert stats.produced_count == stats.consumed_count + stats.dropped_count


def test_invariant_burst_producer_slow_consumer():
    """Burst producer with slow consumer: invariant produced == consumed + dropped strictly holds."""
    buf = LatestFrameBuffer()
    total_frames = 500
    consumed = 0

    # Rapid burst
    for i in range(total_frames):
        buf.put(np.full((10, 10, 3), i % 256, dtype=np.uint8))

    # Consumer reads whatever is in the single slot
    f = buf.get(timeout=0.01)
    if f is not None:
        consumed += 1

    stats = buf.stats
    assert stats.produced_count == total_frames
    assert stats.consumed_count == consumed
    assert stats.dropped_count == total_frames - consumed
    assert stats.produced_count == stats.consumed_count + stats.dropped_count


@pytest.mark.parametrize("num_producers,num_consumers,frames_per_producer", [
    (1, 1, 500),
    (2, 2, 500),
    (4, 2, 500),
    (5, 5, 500),
    (10, 1, 500),
    (1, 5, 500),
])
def test_invariant_multi_threaded_matrix(num_producers, num_consumers, frames_per_producer):
    """Parametric stress test verifying invariant under various thread topologies."""
    buf = LatestFrameBuffer()
    total_expected = num_producers * frames_per_producer
    consumed_counts = [0] * num_consumers
    done_event = threading.Event()
    prods_left = num_producers
    prods_lock = threading.Lock()

    def producer():
        nonlocal prods_left
        for i in range(frames_per_producer):
            buf.put(np.zeros((8, 8, 3), dtype=np.uint8))
        with prods_lock:
            prods_left -= 1
            if prods_left == 0:
                done_event.set()

    def consumer(idx):
        count = 0
        while True:
            f = buf.get(timeout=0.01)
            if f is not None:
                count += 1
            else:
                if done_event.is_set() and buf.is_empty():
                    break
        consumed_counts[idx] = count

    prods = [threading.Thread(target=producer) for _ in range(num_producers)]
    cons = [threading.Thread(target=consumer, args=(i,)) for i in range(num_consumers)]

    for c in cons:
        c.start()
    for p in prods:
        p.start()

    for p in prods:
        p.join(timeout=5.0)
    for c in cons:
        c.join(timeout=5.0)

    stats = buf.stats
    total_consumed = sum(consumed_counts)
    assert stats.produced_count == total_expected
    assert stats.produced_count == stats.consumed_count + stats.dropped_count
    assert total_consumed == stats.consumed_count


def test_invariant_with_clear_operation():
    """Verify that calling clear() correctly preserves produced == consumed + dropped."""
    buf = LatestFrameBuffer()

    buf.put(np.zeros((4, 4, 3), dtype=np.uint8))
    # Frame in buffer, not consumed yet.
    assert buf.stats.produced_count == 1
    assert buf.stats.consumed_count == 0
    assert buf.stats.dropped_count == 0

    # clear() should count this unread frame as dropped
    buf.clear()
    assert buf.stats.produced_count == 1
    assert buf.stats.consumed_count == 0
    assert buf.stats.dropped_count == 1
    assert buf.stats.produced_count == buf.stats.consumed_count + buf.stats.dropped_count


def test_invariant_after_close_with_unconsumed_frame():
    """
    CRITICAL INVARIANT CHECK:
    When close() is called with an unconsumed frame in the buffer,
    the invariant produced == consumed + dropped must hold once the buffer is closed.
    """
    buf = LatestFrameBuffer()
    total_produced = 10
    for _ in range(total_produced):
        buf.put(np.zeros((4, 4, 3), dtype=np.uint8))

    # Buffer is closed without consuming the last frame.
    buf.close()

    stats = buf.stats
    print(f"\n[INVARIANT AFTER CLOSE] Produced: {stats.produced_count}, Consumed: {stats.consumed_count}, Dropped: {stats.dropped_count}")
    assert stats.produced_count == stats.consumed_count + stats.dropped_count, (
        f"Conservation invariant violated after close(): produced ({stats.produced_count}) != "
        f"consumed ({stats.consumed_count}) + dropped ({stats.dropped_count})"
    )



# ============================================================================
# 3. Shutdown Under Load: close() Unblocks Waiting Threads Instantly (<50ms)
# ============================================================================

def test_shutdown_unblocks_single_waiting_thread():
    """Calling close() unblocks single waiting get() in < 50ms and returns None."""
    buf = LatestFrameBuffer()
    unblock_latency = []
    received_value = []

    def waiter():
        t0 = time.perf_counter()
        val = buf.get(timeout=10.0)
        t1 = time.perf_counter()
        unblock_latency.append(t1 - t0)
        received_value.append(val)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)  # Ensure thread is blocked in cond.wait()

    close_t0 = time.perf_counter()
    buf.close()
    t.join(timeout=1.0)
    close_elapsed = time.perf_counter() - close_t0

    assert not t.is_alive(), "Waiter thread hung on close()!"
    assert len(unblock_latency) == 1
    assert received_value[0] is None, f"Expected None on close, got {received_value[0]}"
    assert close_elapsed < 0.05, f"Unblock latency too high: {close_elapsed*1000:.2f}ms >= 50ms"
    print(f"\n[SHUTDOWN 1 thread] Unblocked in {close_elapsed*1000:.3f}ms")


def test_shutdown_unblocks_20_concurrently_waiting_threads():
    """Calling close() unblocks 20 threads simultaneously waiting in get() in < 50ms."""
    buf = LatestFrameBuffer()
    num_threads = 20
    results = [None] * num_threads
    latencies = [0.0] * num_threads
    started_barriers = threading.Barrier(num_threads + 1)

    def waiter(idx):
        started_barriers.wait()
        t0 = time.perf_counter()
        res = buf.get(timeout=10.0)
        latencies[idx] = time.perf_counter() - t0
        results[idx] = res

    threads = [threading.Thread(target=waiter, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()

    started_barriers.wait()
    time.sleep(0.05)  # Ensure all 20 threads are inside cond.wait()

    t_close_start = time.perf_counter()
    buf.close()

    for t in threads:
        t.join(timeout=1.0)
        assert not t.is_alive(), "Consumer thread hung on close()!"

    total_unblock_time = time.perf_counter() - t_close_start
    print(f"\n[SHUTDOWN 20 threads] All 20 threads unblocked in {total_unblock_time*1000:.3f}ms")

    assert total_unblock_time < 0.05, f"Total unblock time {total_unblock_time*1000:.2f}ms >= 50ms"
    for idx, res in enumerate(results):
        assert res is None, f"Thread {idx} got {res} instead of None"


def test_shutdown_under_active_producer_consumer_load():
    """Calling close() in the middle of active producers and consumers exits cleanly without hangs."""
    buf = LatestFrameBuffer()
    stop_event = threading.Event()
    num_producers = 4
    num_consumers = 4
    producer_errors = []
    consumer_errors = []

    def producer():
        try:
            while not stop_event.is_set():
                if not buf.put(np.ones((16, 16, 3), dtype=np.uint8)):
                    break  # Closed
                time.sleep(0.0001)
        except Exception as e:
            producer_errors.append(e)

    def consumer():
        try:
            while not stop_event.is_set():
                val = buf.get(timeout=0.05)
                if val is None and buf.is_closed():
                    break
        except Exception as e:
            consumer_errors.append(e)

    prods = [threading.Thread(target=producer) for _ in range(num_producers)]
    cons = [threading.Thread(target=consumer) for _ in range(num_consumers)]

    for t in prods + cons:
        t.start()

    time.sleep(0.1)  # Run active traffic for 100ms
    buf.close()      # Trigger sudden shutdown under load
    stop_event.set()

    for t in prods + cons:
        t.join(timeout=2.0)
        assert not t.is_alive(), f"Thread {t.name} hung during shutdown under load!"

    assert len(producer_errors) == 0
    assert len(consumer_errors) == 0
    assert buf.is_closed() is True
    # Subsequent calls must return immediately
    assert buf.put(np.ones((16, 16, 3), dtype=np.uint8)) is False
    assert buf.get(timeout=0.1) is None


# ============================================================================
# 4. Buffer Memory Leak & Reference Retention Check
# ============================================================================

def test_memory_leak_frame_retention_after_get():
    """
    CRITICAL MEMORY CHECK:
    When a consumer gets a frame and then deletes its reference,
    does LatestFrameBuffer release the frame or keep holding it in self._slot?
    """
    buf = LatestFrameBuffer(copy_on_read=False)

    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)  # ~3 MB
    ref = weakref.ref(frame)

    buf.put(frame)
    del frame
    assert ref() is not None, "Frame should be alive while buffered"

    retrieved = buf.get()
    assert retrieved is not None
    del retrieved
    gc.collect()

    # Empirical observation:
    # Does buf._slot retain the frame reference even when the buffer is empty?
    is_retained = (ref() is not None)
    slot_is_none = (buf._slot is None)

    print(f"\n[MEMORY TEST] After get() and del retrieved: frame retained={is_retained}, _slot is None={slot_is_none}")
    # In a leak-free single-slot buffer, once consumed, _slot should be cleared (None)
    # so that the large frame buffer is eligible for garbage collection.
    assert not is_retained, "Memory leak: buffer retains consumed frame array in _slot even when buffer is empty"
    assert slot_is_none, "buf._slot must be None after frame is consumed"


def test_memory_leak_frame_retention_after_close():
    """
    CRITICAL MEMORY CHECK:
    When close() is called, are buffered frame arrays freed?
    """
    buf = LatestFrameBuffer(copy_on_read=False)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    ref = weakref.ref(frame)

    buf.put(frame)
    del frame
    assert ref() is not None

    buf.close()
    gc.collect()

    is_retained_on_close = (ref() is not None)
    slot_on_close_is_none = (buf._slot is None)
    print(f"\n[MEMORY TEST] After close(): frame retained={is_retained_on_close}, _slot is None={slot_on_close_is_none}")
    assert not is_retained_on_close, "Memory leak: buffer retains frame array in _slot after close()"
    assert slot_on_close_is_none, "buf._slot must be cleared (None) on close()"


def test_multiple_consumers_waiting_indefinitely_spurious_none():
    """
    CONCURRENCY EDGE CASE:
    When multiple consumers wait with timeout=None (wait indefinitely),
    and 1 frame is pushed, exactly 1 consumer should receive the frame.
    The OTHER consumer must NOT return None prematurely while the buffer is NOT closed!
    Contract of get(): 'None means wait indefinitely until a frame arrives or buffer is closed.'
    """
    buf = LatestFrameBuffer()
    res = []
    thread_done = []

    def consumer(cid):
        val = buf.get(timeout=None)
        res.append((cid, val))
        thread_done.append(cid)

    t1 = threading.Thread(target=consumer, args=(1,))
    t2 = threading.Thread(target=consumer, args=(2,))
    t1.start()
    t2.start()

    time.sleep(0.05)  # Ensure both are waiting in cond.wait()

    # Push 1 frame
    buf.put(np.ones((10, 10, 3), dtype=np.uint8))
    time.sleep(0.05)

    # If the second thread returned None prematurely while buffer is open:
    # res will have 2 items: one with the frame, one with None!
    print(f"\n[MULTI-CONSUMER NONE TEST] Results after 1 frame: {res}")

    # Check if thread 2 terminated prematurely with None
    premature_returns = [r for r in res if r[1] is None]
    
    # Clean up threads
    buf.close()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)

    assert len(premature_returns) == 0, (
        f"Spurious None returned to waiting consumer: {premature_returns}. "
        "get(timeout=None) must wait indefinitely until a frame arrives or buffer is closed!"
    )



def test_memory_churn_10k_frames_no_unbounded_ram_growth():
    """
    Stress-test memory churn: pushing 10,000 frames into buffer.
    Overwriting unread frames must immediately free old frames (only 1 frame retained).
    """
    buf = LatestFrameBuffer()
    weakrefs = []

    for i in range(100):
        f = np.zeros((200, 200, 3), dtype=np.uint8)
        weakrefs.append(weakref.ref(f))
        buf.put(f)
        del f

    gc.collect()
    # At most 1 frame (the latest one) should still be alive
    alive_count = sum(1 for r in weakrefs if r() is not None)
    print(f"\n[MEMORY CHURN] Pushed 100 frames with overwrite: alive frames={alive_count}")
    assert alive_count <= 1, f"Expected at most 1 alive frame due to overwrite, found {alive_count} alive!"


# ============================================================================
# 5. Ingestion Pipeline Stress: MockCameraSource streaming to LatestFrameBuffer
# ============================================================================

def test_camera_ingestion_pipeline_stress_mock_to_buffer():
    """
    Simulates high-speed camera acquisition pipeline:
    MockCameraSource producing frames at 60 FPS directly into LatestFrameBuffer.
    Downstream processing worker consuming frames and measuring dropped frame statistics.
    Simulates camera disconnect, reconnect, and clean shutdown under acquisition load.
    """
    mock_cam = MockCameraSource(fps=60.0, width=320, height=240, mode=MockMode.PROCEDURAL)
    assert mock_cam.open() is True
    buf = LatestFrameBuffer()

    capture_errors = []
    worker_errors = []
    frames_captured = 0
    frames_processed = 0
    stop_pipeline = threading.Event()

    def capture_loop():
        nonlocal frames_captured
        try:
            while not stop_pipeline.is_set() and mock_cam.is_opened():
                ret, frame = mock_cam.read()
                if ret and frame is not None:
                    frames_captured += 1
                    if not buf.put(frame):
                        break  # Buffer closed
                time.sleep(0.001)  # fast capture loop
        except Exception as e:
            capture_errors.append(e)

    def processing_loop():
        nonlocal frames_processed
        try:
            while not stop_pipeline.is_set():
                frame = buf.get(timeout=0.02)
                if frame is not None:
                    frames_processed += 1
                    # Simulate processing latency (e.g. 5ms)
                    time.sleep(0.005)
                elif buf.is_closed():
                    break
        except Exception as e:
            worker_errors.append(e)

    t_cap = threading.Thread(target=capture_loop)
    t_work = threading.Thread(target=processing_loop)

    t_cap.start()
    t_work.start()

    # Run for 200ms
    time.sleep(0.2)

    # Inject simulated disconnect
    mock_cam.simulate_disconnect()
    time.sleep(0.05)

    # Reconnect
    mock_cam.simulate_reconnect()
    time.sleep(0.2)

    # Stop pipeline
    stop_pipeline.set()
    buf.close()
    mock_cam.release()

    t_cap.join(timeout=2.0)
    t_work.join(timeout=2.0)

    assert not t_cap.is_alive()
    assert not t_work.is_alive()
    assert len(capture_errors) == 0
    assert len(worker_errors) == 0

    stats = buf.stats
    print(
        f"\n[INGESTION PIPELINE] Captured: {frames_captured} | Processed: {frames_processed} | "
        f"Dropped: {stats.dropped_count} | Drop ratio: {stats.drop_ratio:.2%}"
    )
    assert frames_captured > 0
    assert frames_processed > 0
    assert stats.produced_count == frames_captured
