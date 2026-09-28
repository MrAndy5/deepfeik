"""Adversarial stress test battery for FaceMeshTracker, CanonicalLandmarks, and mesh geometry.

Tests extreme geometric transformations, boundary clipping, triangle non-degeneracy,
and pathological/extreme image inputs.
"""

from typing import Tuple
import cv2
import numpy as np
import pytest

from deepfeik.core.face_mesh import FaceMeshTracker, FaceBoundingBox, FaceMeshResult
from deepfeik.core.landmarks import (
    CanonicalLandmarks,
    CANONICAL_LANDMARK_INDICES,
    CANONICAL_DELAUNAY_TRIANGLES,
    FACEMESH_FACE_OVAL,
    compute_bounding_box,
    extract_canonical_landmarks,
    get_face_oval_contour,
)
from tests.conftest import create_synthetic_face_image


# ============================================================================
# 1. Extreme Geometric Transformations
# ============================================================================

@pytest.mark.parametrize("angle_deg", [
    -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90
])
def test_adversarial_face_rotation_full_range(angle_deg: int):
    """Synthetic face rotated from -90° to +90° is safely handled without crashing.

    When detected, landmarks and bbox must be finite, non-degenerate, and within frame bounds.
    """
    w, h = 640, 480
    base_face = create_synthetic_face_image(width=w, height=h)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated_frame = cv2.warpAffine(base_face, M, (w, h), borderValue=(40, 40, 40))

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        result = tracker.process(rotated_frame)

    assert isinstance(result, FaceMeshResult)
    if result.has_face:
        # Landmarks must be finite numbers
        assert not np.isnan(result.landmarks_px).any()
        assert not np.isinf(result.landmarks_px).any()

        # Clamped int landmarks must stay strictly within image dimensions
        assert np.all(result.landmarks_int[:, 0] >= 0)
        assert np.all(result.landmarks_int[:, 0] < w)
        assert np.all(result.landmarks_int[:, 1] >= 0)
        assert np.all(result.landmarks_int[:, 1] < h)

        # Bounding box must be valid and non-degenerate
        assert result.bbox is not None
        assert 0 <= result.bbox.x1 <= result.bbox.x2 <= w
        assert 0 <= result.bbox.y1 <= result.bbox.y2 <= h
        assert result.bbox.width > 0
        assert result.bbox.height > 0
        assert result.bbox.area > 0

        # Canonical landmarks must have shape (130, 2)
        assert result.canonical_px is not None
        assert result.canonical_px.shape == (130, 2)


@pytest.mark.parametrize("stretch_name, nw, nh", [
    ("horizontal_3x", 1920, 480),
    ("vertical_3x", 640, 1440),
    ("uniform_scale_3x", 1920, 1440),
    ("uniform_compress_3x", 213, 160),
])
def test_adversarial_face_stretching(stretch_name: str, nw: int, nh: int):
    """Extreme 3x stretching and 0.33x compression are handled safely without crashing."""
    base_face = create_synthetic_face_image(width=640, height=480)
    stretched = cv2.resize(base_face, (nw, nh))

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        result = tracker.process(stretched)

    assert isinstance(result, FaceMeshResult)
    if result.has_face:
        assert result.bbox is not None
        assert 0 <= result.bbox.x1 <= result.bbox.x2 <= nw
        assert 0 <= result.bbox.y1 <= result.bbox.y2 <= nh
        assert result.bbox.width > 0
        assert result.bbox.height > 0


def test_adversarial_face_horizontal_flip():
    """Horizontally mirrored face is successfully detected and preserves mesh geometry."""
    base_face = create_synthetic_face_image(width=640, height=480)
    flipped = cv2.flip(base_face, 1)

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        result = tracker.process(flipped)

    assert result.has_face is True
    assert result.bbox is not None
    assert result.bbox.width > 100
    assert result.bbox.height > 100

    # Delaunay triangles must remain non-degenerate on flipped face
    canonical = result.canonical_px
    tri = CANONICAL_DELAUNAY_TRIANGLES
    v1, v2, v3 = canonical[tri[:, 0]], canonical[tri[:, 1]], canonical[tri[:, 2]]
    areas = 0.5 * np.abs((v2[:, 0] - v1[:, 0]) * (v3[:, 1] - v1[:, 1]) - (v3[:, 0] - v1[:, 0]) * (v2[:, 1] - v1[:, 1]))
    assert np.all(areas > 0.0)


def test_adversarial_face_upside_down():
    """Upside-down (180° rotated) face does not crash and safely returns no face."""
    base_face = create_synthetic_face_image(width=640, height=480)
    upside_down = cv2.flip(base_face, 0)

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        result = tracker.process(upside_down)

    assert isinstance(result, FaceMeshResult)
    # MediaPipe naturally rejects upside down face orientation without crashing
    assert result.has_face is False


# ============================================================================
# 2. Boundary Clipping and Integer Clamping
# ============================================================================

@pytest.mark.parametrize("name, dx, dy", [
    ("50pct_left", -100, 0),
    ("50pct_right", 100, 0),
    ("50pct_up", 0, -100),
    ("50pct_down", 0, 100),
    ("extreme_off_left", -250, 0),
    ("extreme_off_right", 250, 0),
    ("extreme_off_up", 0, -200),
    ("extreme_off_down", 0, 200),
    ("completely_off_left", -600, 0),
    ("completely_off_right", 600, 0),
])
def test_adversarial_boundary_clipping(name: str, dx: int, dy: int):
    """Face shifted partially or completely outside frame maintains strict coordinate bounds."""
    w, h = 640, 480
    base_face = create_synthetic_face_image(width=w, height=h)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(base_face, M, (w, h), borderValue=(40, 40, 40))

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        result = tracker.process(shifted)

    assert isinstance(result, FaceMeshResult)
    if result.has_face:
        # Landmarks int must be clamped to [0, w-1] and [0, h-1]
        assert np.all(result.landmarks_int[:, 0] >= 0)
        assert np.all(result.landmarks_int[:, 0] < w)
        assert np.all(result.landmarks_int[:, 1] >= 0)
        assert np.all(result.landmarks_int[:, 1] < h)

        # Bounding box must be clamped strictly within [0, w] and [0, h]
        assert result.bbox is not None
        assert 0 <= result.bbox.x1 <= result.bbox.x2 <= w
        assert 0 <= result.bbox.y1 <= result.bbox.y2 <= h
        assert result.bbox.width >= 10
        assert result.bbox.height >= 10

        # roi_slice indexing must be safe and valid on a frame
        roi = shifted[result.bbox.roi_slice]
        assert roi.shape[0] == result.bbox.height
        assert roi.shape[1] == result.bbox.width


def test_adversarial_compute_bounding_box_pathological_coordinates():
    """compute_bounding_box safely clamps pathological out-of-frame coordinates."""
    w, h = 640, 480
    frame_shape = (h, w)
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. 50%+ coordinates outside normalized frame (negative and > 1.0)
    rng = np.random.default_rng(42)
    mock_lms_norm = rng.uniform(-1.0, 2.0, (468, 2)).astype(np.float32)
    mock_lms_px = mock_lms_norm.copy()
    mock_lms_px[:, 0] *= w
    mock_lms_px[:, 1] *= h

    bbox = compute_bounding_box(mock_lms_px, frame_shape=frame_shape, margin=0.15)
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 <= x2 <= w
    assert 0 <= y1 <= y2 <= h

    # 2. All coordinates completely negative (verify clamped to [0, W] and [0, H], no negative output)
    all_neg = rng.uniform(-500.0, -50.0, (468, 2)).astype(np.float32)
    bbox_neg = compute_bounding_box(all_neg, frame_shape=frame_shape, margin=0.15)
    x1, y1, x2, y2 = bbox_neg
    assert 0 <= x1 <= x2 <= w
    assert 0 <= y1 <= y2 <= h
    # Verify FaceBoundingBox construction and slice does not crash
    fb_neg = FaceBoundingBox(*bbox_neg)
    assert fb_neg.width >= 0
    assert fb_neg.height >= 0
    assert len(frame[fb_neg.roi_slice]) >= 0

    # 3. All coordinates completely beyond frame bounds
    all_pos = rng.uniform(800.0, 1500.0, (468, 2)).astype(np.float32)
    bbox_pos = compute_bounding_box(all_pos, frame_shape=frame_shape, margin=0.15)
    x1, y1, x2, y2 = bbox_pos
    assert 0 <= x1 <= x2 <= w
    assert 0 <= y1 <= y2 <= h
    fb_pos = FaceBoundingBox(*bbox_pos)
    assert fb_pos.width >= 0
    assert fb_pos.height >= 0


def test_adversarial_face_bounding_box_invariants():
    """FaceBoundingBox maintains non-negative dimensions and safe slices for degenerate boxes."""
    # Inverted box coordinates
    bbox_inv = FaceBoundingBox(x1=100, y1=100, x2=50, y2=50)
    assert bbox_inv.width == 0
    assert bbox_inv.height == 0
    assert bbox_inv.area == 0
    r_slice, c_slice = bbox_inv.roi_slice
    assert r_slice == slice(100, 50)
    assert c_slice == slice(100, 50)

    # Empty frame slice does not crash
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cropped = frame[r_slice, c_slice]
    assert cropped.size == 0


# ============================================================================
# 3. Triangle Validation and Topology Invariants
# ============================================================================

def test_adversarial_canonical_triangles_inventory():
    """CANONICAL_DELAUNAY_TRIANGLES has exactly 221 simplices indexing [0, 129]."""
    triangles = CANONICAL_DELAUNAY_TRIANGLES
    assert triangles.shape == (221, 3)
    assert triangles.dtype == np.int32

    # Verify all indices are strictly within [0, 129]
    min_idx = int(np.min(triangles))
    max_idx = int(np.max(triangles))
    assert min_idx == 0, f"Expected min index 0, got {min_idx}"
    assert max_idx == 129, f"Expected max index 129, got {max_idx}"

    # Verify canonical indices array has length 130
    assert len(CANONICAL_LANDMARK_INDICES) == 130
    assert len(np.unique(CANONICAL_LANDMARK_INDICES)) == 130


def test_adversarial_canonical_triangles_no_duplicate_vertices():
    """Every triangle in CANONICAL_DELAUNAY_TRIANGLES consists of 3 distinct vertex indices."""
    for i, t in enumerate(CANONICAL_DELAUNAY_TRIANGLES):
        distinct = set(t)
        assert len(distinct) == 3, f"Triangle index {i} has degenerate duplicate vertices: {t}"


def test_adversarial_canonical_triangles_non_degenerate_area():
    """All 221 static triangles have non-zero positive area on a canonical face."""
    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        res = tracker.process(create_synthetic_face_image())

    assert res.has_face is True
    canonical = res.canonical_px
    tri = CANONICAL_DELAUNAY_TRIANGLES

    v1 = canonical[tri[:, 0]]
    v2 = canonical[tri[:, 1]]
    v3 = canonical[tri[:, 2]]

    # 2D cross product for signed triangle area: 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    cross_product = (v2[:, 0] - v1[:, 0]) * (v3[:, 1] - v1[:, 1]) - (v3[:, 0] - v1[:, 0]) * (v2[:, 1] - v1[:, 1])
    signed_areas = 0.5 * cross_product
    abs_areas = np.abs(signed_areas)

    # Assert strictly no zero-area triangles
    zero_area_count = int(np.count_nonzero(abs_areas == 0.0))
    assert zero_area_count == 0, f"Found {zero_area_count} degenerate triangles with area == 0"

    # Assert minimal area threshold (no sliver triangles collapsed to near-zero)
    min_area = float(np.min(abs_areas))
    assert min_area > 1.0, f"Triangle minimum area {min_area} px^2 is too small (< 1.0 px^2)"

    # Assert all 221 triangles share uniform positive winding order
    positive_winding_count = int(np.count_nonzero(signed_areas > 0))
    assert positive_winding_count == 221, (
        f"Inconsistent winding orders: {positive_winding_count} positive, {221 - positive_winding_count} negative"
    )


def test_adversarial_canonical_triangles_caching_and_idempotency():
    """CanonicalLandmarks.get_static_triangles returns cached topology without recomputation."""
    t1 = CanonicalLandmarks.get_static_triangles()
    assert t1 is CANONICAL_DELAUNAY_TRIANGLES
    assert len(t1) == 221


# ============================================================================
# 4. Extreme and Pathological Inputs
# ============================================================================

def test_adversarial_extreme_pure_black_frame():
    """Pure black frame (all 0s) is handled cleanly returning has_face=False without crashing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res = tracker.process(frame)
    assert res.has_face is False
    assert res.bbox is None
    assert res.landmarks_px is None


def test_adversarial_extreme_pure_white_frame():
    """Pure white frame (all 255s) is handled cleanly returning has_face=False without crashing."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res = tracker.process(frame)
    assert res.has_face is False
    assert res.bbox is None
    assert res.landmarks_px is None


def test_adversarial_extreme_gaussian_noise_frame():
    """Gaussian random noise image does not trigger false positive detections or crashes."""
    rng = np.random.default_rng(999)
    noise = rng.normal(128, 50, (480, 640, 3)).clip(0, 255).astype(np.uint8)
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res = tracker.process(noise)
    assert res.has_face is False
    assert res.bbox is None


def test_adversarial_extreme_single_pixel_frame():
    """Single-pixel frame (1x1x3) is handled gracefully without crashing."""
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res = tracker.process(frame)
    assert res.has_face is False
    assert res.bbox is None


def test_adversarial_extreme_4k_resolution_blank():
    """4K resolution blank frame (4096x2160x3) executes cleanly without out-of-memory or crash."""
    frame = np.zeros((2160, 4096, 3), dtype=np.uint8)
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res = tracker.process(frame)
    assert res.has_face is False
    assert res.bbox is None


def test_adversarial_extreme_4k_resolution_synthetic_face():
    """4K resolution frame with synthetic face detects face and produces valid coordinates."""
    frame = create_synthetic_face_image(width=4096, height=2160, axes=(500, 700))
    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        res = tracker.process(frame)

    assert res.has_face is True
    assert res.bbox is not None
    assert 0 <= res.bbox.x1 <= res.bbox.x2 <= 4096
    assert 0 <= res.bbox.y1 <= res.bbox.y2 <= 2160
    assert res.bbox.width > 500
    assert res.bbox.height > 500

    # Ensure all landmarks_int are bounded in 4K resolution
    assert np.all(result_x >= 0 for result_x in res.landmarks_int[:, 0])
    assert np.all(result_x < 4096 for result_x in res.landmarks_int[:, 0])
    assert np.all(result_y >= 0 for result_y in res.landmarks_int[:, 1])
    assert np.all(result_y < 2160 for result_y in res.landmarks_int[:, 1])


def test_adversarial_extreme_aspect_ratios():
    """Extreme aspect ratios (1x1000 and 1000x1) are rejected safely without crashing."""
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res_wide = tracker.process(np.zeros((1, 1000, 3), dtype=np.uint8))
        assert res_wide.has_face is False

        res_tall = tracker.process(np.zeros((1000, 1, 3), dtype=np.uint8))
        assert res_tall.has_face is False


def test_adversarial_non_contiguous_and_read_only_arrays():
    """Tracker accepts non-contiguous (strided) and read-only NumPy array buffers."""
    base_face = create_synthetic_face_image(width=640, height=480)

    # 1. Non-contiguous buffer (negative stride)
    non_contig = base_face[:, ::-1, :]
    assert not non_contig.flags["C_CONTIGUOUS"]

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        res_nc = tracker.process(non_contig)
        assert res_nc.has_face is True

    # 2. Read-only buffer
    read_only = base_face.copy()
    read_only.setflags(write=False)
    assert not read_only.flags["WRITEABLE"]

    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        res_ro = tracker.process(read_only)
        assert res_ro.has_face is True


def test_adversarial_zero_dimension_frame_shapes():
    """FaceMeshTracker handles non-positive height or width frames cleanly."""
    with FaceMeshTracker(static_image_mode=True) as tracker:
        res_zero_w = tracker.process(np.zeros((100, 0, 3), dtype=np.uint8))
        assert res_zero_w.has_face is False

        res_zero_h = tracker.process(np.zeros((0, 100, 3), dtype=np.uint8))
        assert res_zero_h.has_face is False


def test_adversarial_canonical_landmarks_utility_coverage():
    """CanonicalLandmarks class methods and invalid inputs."""
    indices = CanonicalLandmarks.get_canonical_indices()
    assert len(indices) == 130

    # compute_delaunay_triangles with filter_inside_oval=False
    pts = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    tri = CanonicalLandmarks.compute_delaunay_triangles(pts, filter_inside_oval=False)
    assert len(tri) >= 1

    # get_face_oval_contour with invalid landmark count
    with pytest.raises(ValueError):
        get_face_oval_contour(np.zeros((50, 2), dtype=np.float32))
