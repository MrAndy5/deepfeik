"""
Unit tests for FaceSwapEngine and CPU throughput benchmark (>= 15 FPS).
"""

import time
import cv2
import numpy as np
import pytest

from deepfeik.core.engine import FaceSwapEngine
from tests.conftest import create_synthetic_face_image, create_blank_image


@pytest.fixture
def engine():
    """Yields a clean FaceSwapEngine instance."""
    eng = FaceSwapEngine()
    yield eng
    eng.close()


def test_engine_initialization_and_clear(engine):
    """Verifies initial state is unloaded and clearing resets state."""
    assert engine.is_source_loaded() is False

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    out = engine.process_frame(frame)
    assert np.array_equal(out, frame), "Must pass through when unloaded"

    engine.clear_source_face()
    assert engine.is_source_loaded() is False


def test_engine_set_source_valid(engine):
    """Verifies loading a valid synthetic face image succeeds."""
    face = create_synthetic_face_image(640, 480)
    success, msg = engine.set_source_face(face)
    assert success is True
    assert "success" in msg.lower()
    assert engine.is_source_loaded() is True


def test_engine_set_source_no_face(engine):
    """Verifies attempting to load a non-face image fails with informative message."""
    blank = create_blank_image(640, 480)
    success, msg = engine.set_source_face(blank)
    assert success is False
    assert "face" in msg.lower()
    assert engine.is_source_loaded() is False


def test_engine_set_source_channels(engine):
    """Verifies loading 1-channel grayscale and 4-channel BGRA images."""
    bgr = create_synthetic_face_image(640, 480)

    # 1. Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    success, msg = engine.set_source_face(gray)
    assert success is True
    assert engine.is_source_loaded() is True

    # 2. 4-Channel BGRA
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    success, msg = engine.set_source_face(bgra)
    assert success is True
    assert engine.is_source_loaded() is True


def test_engine_process_frame_active_swap(engine):
    """Verifies face swapping modifies facial ROI while keeping background intact."""
    src = create_synthetic_face_image(640, 480, skin_color=(140, 160, 220))
    tgt = create_synthetic_face_image(640, 480, skin_color=(180, 160, 140))

    engine.set_source_face(src)
    assert engine.is_source_loaded() is True

    swapped = engine.process_frame(tgt)
    assert swapped is not None
    assert swapped.shape == tgt.shape

    # Center of face should be modified
    h, w = tgt.shape[:2]
    center_tgt = tgt[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_swapped = swapped[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    assert not np.array_equal(center_tgt, center_swapped)

    # Background corners should remain identical
    assert np.allclose(swapped[0:15, 0:15], tgt[0:15, 0:15])


def test_engine_resolution_switch(engine):
    """Verifies engine dynamically handles live resolution switches."""
    src = create_synthetic_face_image(640, 480)
    engine.set_source_face(src)

    # 640x480 frame
    f1 = create_synthetic_face_image(640, 480)
    out1 = engine.process_frame(f1)
    assert out1.shape == (480, 640, 3)

    # 800x600 frame
    f2 = create_synthetic_face_image(800, 600)
    out2 = engine.process_frame(f2)
    assert out2.shape == (600, 800, 3)


def test_engine_fps_benchmark(engine):
    """Verifies that FaceSwapEngine.process_frame() achieves >= 15 FPS on CPU."""
    src = create_synthetic_face_image(640, 480)
    tgt = create_synthetic_face_image(640, 480)

    engine.set_source_face(src)

    # Warmup
    for _ in range(3):
        _ = engine.process_frame(tgt)

    n_frames = 20
    t0 = time.perf_counter()
    for _ in range(n_frames):
        _ = engine.process_frame(tgt)
    total_time = time.perf_counter() - t0

    fps = n_frames / total_time
    avg_latency_ms = (total_time / n_frames) * 1000.0

    print(f"\nFaceSwapEngine CPU Performance: {fps:.1f} FPS (Mean Latency: {avg_latency_ms:.2f} ms/frame)")
    assert fps >= 15.0, f"CPU FPS too low: {fps:.1f} < 15.0 (Latency: {avg_latency_ms:.2f} ms)"


def test_engine_context_manager():
    """Verifies FaceSwapEngine works as a context manager."""
    with FaceSwapEngine() as eng:
        assert eng.is_source_loaded() is False
