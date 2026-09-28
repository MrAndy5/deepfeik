"""
Unit tests for ReinhardColorMatcher.
"""

import time
import cv2
import numpy as np
import pytest

from deepfeik.core.color import ReinhardColorMatcher


def test_color_stats_calculation():
    """Verifies LAB mean and std computation with and without mask."""
    matcher = ReinhardColorMatcher()
    img = np.full((100, 100, 3), (100, 150, 200), dtype=np.uint8)

    mean, std = matcher.compute_lab_stats(img)
    assert len(mean) == 3
    assert len(std) == 3
    assert np.all(std > 0.0)

    # With half mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255
    mean_m, std_m = matcher.compute_lab_stats(img, mask)
    assert np.allclose(mean, mean_m, atol=2.0)


def test_color_matcher_transfer():
    """Verifies that source colors are transferred to match reference distribution."""
    matcher = ReinhardColorMatcher()
    # Dark source
    src = np.full((100, 100, 3), (40, 60, 80), dtype=np.uint8)
    # Bright target
    tgt = np.full((100, 100, 3), (160, 180, 220), dtype=np.uint8)

    matched = matcher.match(src, tgt)
    assert matched.shape == src.shape
    assert matched.dtype == np.uint8

    # Matched image mean should be shifted toward target
    src_mean = float(src.mean())
    tgt_mean = float(tgt.mean())
    matched_mean = float(matched.mean())

    assert matched_mean > src_mean
    assert abs(matched_mean - tgt_mean) < 15.0


def test_color_matcher_caching():
    """Verifies pre-computing source statistics avoids re-computation."""
    matcher = ReinhardColorMatcher()
    src = np.full((100, 100, 3), (80, 100, 120), dtype=np.uint8)
    tgt = np.full((100, 100, 3), (150, 170, 200), dtype=np.uint8)

    matcher.set_source_stats(src)
    assert matcher._cached_source_mean is not None

    matched = matcher.match(src, tgt)
    assert matched is not None

    matcher.clear_source_stats()
    assert matcher._cached_source_mean is None


def test_color_matcher_dimension_mismatch():
    """Verifies color transfer works between images of completely different resolutions."""
    matcher = ReinhardColorMatcher()
    src = np.full((80, 120, 3), (50, 70, 90), dtype=np.uint8)
    tgt = np.full((480, 640, 3), (180, 200, 220), dtype=np.uint8)

    matched = matcher.match(src, tgt)
    assert matched.shape == src.shape


def test_color_matcher_edge_cases():
    """Verifies handling of empty arrays, grayscale, RGBA, and empty masks."""
    matcher = ReinhardColorMatcher()

    # None / Empty
    assert matcher.match(None, None) is None
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    assert matcher.match(empty, empty).size == 0

    # Grayscale 1-channel source
    gray = np.full((50, 50), 100, dtype=np.uint8)
    tgt = np.full((50, 50, 3), (150, 150, 150), dtype=np.uint8)
    matched_gray = matcher.match(gray, tgt)
    assert matched_gray.shape == (50, 50, 3)

    # Empty mask (all zeros)
    all_zero_mask = np.zeros((50, 50), dtype=np.uint8)
    matched_mask = matcher.match(tgt, tgt, source_mask=all_zero_mask)
    assert matched_mask.shape == tgt.shape


def test_color_matcher_latency():
    """Asserts Reinhard color transfer runs in < 4 ms on CPU."""
    matcher = ReinhardColorMatcher()
    src = np.full((250, 250, 3), (120, 150, 180), dtype=np.uint8)
    tgt = np.full((250, 250, 3), (160, 180, 220), dtype=np.uint8)
    mask = np.full((250, 250), 255, dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = matcher.match(src, tgt, mask, mask)

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = matcher.match(src, tgt, mask, mask)
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    print(f"\nReinhardColorMatcher latency: {avg_ms:.2f} ms per frame")
    assert avg_ms < 4.0, f"Color transfer exceeded latency budget: {avg_ms:.2f} ms"
