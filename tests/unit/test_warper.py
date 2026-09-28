"""
Unit tests for PiecewiseAffineWarper.
"""

import time
import cv2
import numpy as np
import pytest

from deepfeik.core.landmarks import CANONICAL_DELAUNAY_TRIANGLES, CanonicalLandmarks
from deepfeik.core.warper import PiecewiseAffineWarper
from tests.conftest import create_synthetic_face_image
from deepfeik.core.face_mesh import FaceMeshTracker


@pytest.fixture(scope="module")
def sample_faces():
    """Generates source and target synthetic face images and their landmarks."""
    src_img = create_synthetic_face_image(640, 480, skin_color=(150, 180, 240))
    tgt_img = create_synthetic_face_image(640, 480, skin_color=(180, 160, 140))
    with FaceMeshTracker(static_mode=True) as tracker:
        res_src = tracker.process(src_img)
        res_tgt = tracker.process(tgt_img)
    return src_img, res_src.canonical_px, res_src.landmarks_px, tgt_img, res_tgt.canonical_px, res_tgt.landmarks_px


def test_warper_initialization():
    """Verifies warper initializes with canonical or custom triangles."""
    warper = PiecewiseAffineWarper()
    assert len(warper.triangles) == len(CANONICAL_DELAUNAY_TRIANGLES)
    assert warper.has_cached_source() is False

    custom_triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    warper_custom = PiecewiseAffineWarper(triangles=custom_triangles)
    assert len(warper_custom.triangles) == 2


def test_warper_caching(sample_faces):
    """Verifies source pre-caching, clearing, and state tracking."""
    src_img, src_canonical, _, _, _, _ = sample_faces
    warper = PiecewiseAffineWarper()

    warper.set_source(src_img, src_canonical)
    assert warper.has_cached_source() is True

    warper.clear_source()
    assert warper.has_cached_source() is False

    with pytest.raises(RuntimeError):
        warper.warp_to_target(src_canonical, (480, 640))


def test_warper_warp_to_target(sample_faces):
    """Verifies warping pre-cached source produces valid non-zero warped face."""
    src_img, src_canonical, _, tgt_img, tgt_canonical, _ = sample_faces
    warper = PiecewiseAffineWarper()
    warper.set_source(src_img, src_canonical)

    warped = warper.warp_to_target(tgt_canonical, (480, 640))
    assert warped is not None
    assert warped.shape == (480, 640, 3)
    assert warped.dtype == np.uint8

    # Non-zero face pixels
    non_zero = np.count_nonzero(warped)
    assert non_zero > 10000, f"Expected substantial warped face pixels, got {non_zero}"


def test_warper_dense_landmarks_handling(sample_faces):
    """Verifies warper accepts dense landmarks (>=468 points) and auto-subsamples."""
    src_img, _, src_dense, tgt_img, _, tgt_dense = sample_faces
    warper = PiecewiseAffineWarper()

    warped = warper.warp(src_img, src_dense, tgt_dense, (480, 640))
    assert warped.shape == (480, 640, 3)
    assert np.count_nonzero(warped) > 10000


def test_warper_out_of_bounds_handling(sample_faces):
    """Verifies warper handles landmarks outside image canvas without crash."""
    src_img, src_canonical, _, _, tgt_canonical, _ = sample_faces
    warper = PiecewiseAffineWarper()
    warper.set_source(src_img, src_canonical)

    # Shift target landmarks partially outside canvas (negative and beyond boundary)
    shifted_target = tgt_canonical - np.array([200.0, 150.0], dtype=np.float32)
    warped = warper.warp_to_target(shifted_target, (480, 640))
    assert warped.shape == (480, 640, 3)


def test_warper_degenerate_triangles():
    """Verifies collinear or zero-size triangles are skipped safely."""
    # Triangles with degenerate indices
    degenerate_triangles = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int32)
    warper = PiecewiseAffineWarper(triangles=degenerate_triangles)

    pts = np.array([[10, 10], [10, 10], [10, 10], [20, 20]], dtype=np.float32)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    warper.set_source(img, pts)
    warped = warper.warp_to_target(pts, (100, 100))
    assert warped.shape == (100, 100, 3)


def test_warper_latency(sample_faces):
    """Asserts warping 221 triangles runs in < 25 ms on CPU (real-world < 10 ms;
    budget is relaxed to avoid flakiness under test-environment load)."""
    src_img, src_canonical, _, _, tgt_canonical, _ = sample_faces
    warper = PiecewiseAffineWarper()
    warper.set_source(src_img, src_canonical)

    # Warmup
    for _ in range(3):
        _ = warper.warp_to_target(tgt_canonical, (480, 640))

    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = warper.warp_to_target(tgt_canonical, (480, 640))
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    print(f"\nPiecewiseAffineWarper latency: {avg_ms:.2f} ms per frame")
    assert avg_ms < 25.0, f"Warper exceeded latency budget: {avg_ms:.2f} ms"
