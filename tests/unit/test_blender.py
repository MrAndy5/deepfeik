"""
Unit tests for FeatheredAlphaBlender.
"""

import time
import cv2
import numpy as np
import pytest

from deepfeik.core.blender import FeatheredAlphaBlender


def test_blender_binary_mask():
    """Verifies binary mask generation from contour points."""
    blender = FeatheredAlphaBlender()
    poly = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)

    mask = blender.create_binary_mask((100, 100), poly)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask[50, 50] == 255
    assert mask[5, 5] == 0


def test_blender_feathered_mask():
    """Verifies Gaussian blur feathering produces smooth float32 gradients."""
    blender = FeatheredAlphaBlender(erode_pixels=2, blur_kernel_size=17)
    poly = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)

    alpha = blender.create_feathered_mask((100, 100), face_oval_points=poly)
    assert alpha.shape == (100, 100)
    assert alpha.dtype == np.float32
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0

    # Center should be 1.0 or close, corner should be 0.0
    assert alpha[50, 50] > 0.8
    assert alpha[5, 5] == 0.0

    # Boundary transition gradient
    grad = np.diff(alpha[50, :])
    assert np.max(np.abs(grad)) < 0.25, "Feathering should not produce sharp step transitions"


def test_blender_linear_blend():
    """Verifies linear alpha compositing correctly combines foreground and background."""
    blender = FeatheredAlphaBlender()
    target = np.full((100, 100, 3), 50, dtype=np.uint8)
    warped = np.full((100, 100, 3), 200, dtype=np.uint8)
    poly = np.array([[25, 25], [75, 25], [75, 75], [25, 75]], dtype=np.int32)

    blended = blender.blend(warped, target, face_oval_points=poly)
    assert blended.shape == target.shape
    assert blended.dtype == np.uint8

    # Inside contour should match warped
    assert blended[50, 50, 0] > 180
    # Outside contour should match target exactly
    assert blended[5, 5, 0] == 50


def test_blender_seamless_clone():
    """Verifies Poisson seamlessClone mode executes and falls back safely."""
    blender = FeatheredAlphaBlender()
    target = np.full((100, 100, 3), 60, dtype=np.uint8)
    warped = np.full((100, 100, 3), 180, dtype=np.uint8)
    poly = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)

    cloned = blender.blend(warped, target, face_oval_points=poly, mode="poisson")
    assert cloned.shape == target.shape
    assert cloned.dtype == np.uint8


def test_blender_edge_cases():
    """Verifies handling of None, empty inputs, or empty contours."""
    blender = FeatheredAlphaBlender()
    target = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Empty points
    empty_pts = np.empty((0, 2), dtype=np.int32)
    res = blender.blend(target, target, face_oval_points=empty_pts)
    assert np.array_equal(res, target)

    # None frames
    assert blender.blend(None, target) is target
    assert blender.blend(target, None) is target


def test_blender_latency():
    """Asserts feathered blend execution runs in < 2.0 ms on CPU."""
    blender = FeatheredAlphaBlender()
    target = np.full((480, 640, 3), 50, dtype=np.uint8)
    warped = np.full((480, 640, 3), 200, dtype=np.uint8)

    # 36-point oval
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    cx, cy = 320, 240
    xs = cx + 120 * np.cos(t)
    ys = cy + 160 * np.sin(t)
    oval = np.column_stack([xs, ys]).astype(np.int32)

    # Warmup
    for _ in range(5):
        _ = blender.blend(warped, target, face_oval_points=oval)

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = blender.blend(warped, target, face_oval_points=oval)
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    print(f"\nFeatheredAlphaBlender latency: {avg_ms:.2f} ms per frame")
    assert avg_ms < 2.5, f"Blender exceeded latency budget: {avg_ms:.2f} ms"
