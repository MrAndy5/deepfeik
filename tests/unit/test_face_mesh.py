"""Unit tests for FaceMeshTracker, FaceBoundingBox, and CanonicalLandmarks."""
import cv2
import numpy as np
import pytest

from deepfeik.core.face_mesh import FaceMeshTracker, FaceBoundingBox, FaceMeshResult
from deepfeik.core.landmarks import (
    CanonicalLandmarks,
    CANONICAL_LANDMARK_INDICES,
    FACEMESH_FACE_OVAL,
    CANONICAL_DELAUNAY_TRIANGLES,
    extract_canonical_landmarks,
    compute_bounding_box,
    get_face_oval_contour,
)


# ============================================================================
# 1. FaceMeshTracker Tests
# ============================================================================

def test_tracker_initialization_modes():
    """FaceMeshTracker initializes cleanly in both video and static image modes."""
    tracker_video = FaceMeshTracker(static_image_mode=False)
    assert tracker_video is not None
    tracker_video.close()

    tracker_static = FaceMeshTracker(static_image_mode=True)
    assert tracker_static is not None
    tracker_static.close()


def test_tracker_detect_synthetic_face_success(synthetic_face_image):
    """Tracker successfully detects face in synthetic face image."""
    tracker = FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1)
    landmarks, bbox = tracker.process_frame(synthetic_face_image)
    tracker.close()

    assert landmarks is not None, "Failed to detect face on procedural face image"
    assert bbox is not None
    assert isinstance(landmarks, np.ndarray)
    assert landmarks.shape[0] in (468, 478)
    assert landmarks.shape[1] in (2, 3)

    # Check finite numbers and coordinate bounds
    assert not np.isnan(landmarks).any()
    assert not np.isinf(landmarks).any()


def test_tracker_detect_failure_on_blank_image(blank_black_image):
    """Tracker returns None for blank all-black image with 0 faces."""
    tracker = FaceMeshTracker(static_image_mode=True)
    landmarks, bbox = tracker.process_frame(blank_black_image)
    tracker.close()

    assert landmarks is None
    assert bbox is None


def test_tracker_detect_failure_on_random_noise(random_noise_image):
    """Tracker returns None for uniform random noise image without false positives."""
    tracker = FaceMeshTracker(static_image_mode=True)
    landmarks, bbox = tracker.process_frame(random_noise_image)
    tracker.close()

    assert landmarks is None
    assert bbox is None


def test_tracker_context_manager(synthetic_face_image):
    """FaceMeshTracker supports context manager protocol with automatic cleanup."""
    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        landmarks, _ = tracker.process_frame(synthetic_face_image)
        assert landmarks is not None


def test_tracker_consecutive_frames_stability(synthetic_face_image):
    """Tracker maintains stable landmark coordinates across consecutive video frames."""
    with FaceMeshTracker(static_image_mode=False, min_detection_confidence=0.1) as tracker:
        prev_coords = None
        for _ in range(5):
            landmarks, _ = tracker.process_frame(synthetic_face_image)
            assert landmarks is not None
            if prev_coords is not None:
                max_jitter = np.max(np.abs(landmarks[:, :2] - prev_coords[:, :2]))
                assert max_jitter < 5.0, "Excessive jitter across identical video frames"
            prev_coords = landmarks.copy()


def test_face_mesh_result_dataclass_structure(synthetic_face_image):
    """FaceMeshTracker.process() returns rich structured FaceMeshResult."""
    with FaceMeshTracker(static_mode=True, min_detection_confidence=0.1) as tracker:
        result = tracker.process(synthetic_face_image)
        assert isinstance(result, FaceMeshResult)
        assert result.has_face is True
        assert result.landmarks_norm is not None
        assert result.landmarks_px is not None
        assert result.landmarks_int is not None
        assert result.canonical_px is not None
        assert result.bbox is not None
        assert result.face_oval is not None
        assert result.convex_hull is not None

        assert result.landmarks_px.shape[0] in (468, 478)
        assert result.canonical_px.shape == (130, 2)
        assert result.face_oval.shape == (36, 2)
        assert isinstance(result.bbox, FaceBoundingBox)
        assert result.bbox.width > 50 and result.bbox.height > 50
        assert result.bbox.area > 2500
        assert len(result.bbox.roi_slice) == 2


def test_face_mesh_edge_cases():
    """FaceMeshTracker handles null, empty, and degenerate input frames safely."""
    with FaceMeshTracker(static_mode=False) as tracker:
        # None frame
        assert tracker.process(None).has_face is False
        assert tracker.process_frame(None) == (None, None)
        # Empty frame
        assert tracker.process(np.empty((0, 0, 3), dtype=np.uint8)).has_face is False
        # Black frame
        assert tracker.process(np.zeros((480, 640, 3), dtype=np.uint8)).has_face is False


# ============================================================================
# 2. CanonicalLandmarks & Geometric Primitives Tests
# ============================================================================

def test_canonical_indices_definitions():
    """CANONICAL_LANDMARK_INDICES contains ~120-140 unique valid landmark indices."""
    indices = np.array(CANONICAL_LANDMARK_INDICES)
    assert 115 <= len(indices) <= 145, f"Expected ~130 indices, got {len(indices)}"
    assert np.all(indices >= 0)
    assert np.all(indices < 468)
    assert len(indices) == len(np.unique(indices)), "Canonical indices must be unique"


def test_face_oval_indices_exact_count():
    """FACEMESH_FACE_OVAL contains exactly 36 contour perimeter indices."""
    assert len(FACEMESH_FACE_OVAL) == 36
    assert np.all(np.isin(FACEMESH_FACE_OVAL, CANONICAL_LANDMARK_INDICES)), (
        "All Face Oval indices must be included in CANONICAL_LANDMARK_INDICES"
    )


def test_extract_canonical_landmarks():
    """extract_canonical_landmarks downsamples 468 dense points to canonical subset."""
    dense_pts = np.arange(468 * 2, dtype=np.float32).reshape(468, 2)
    canonical = extract_canonical_landmarks(dense_pts)

    assert canonical.shape == (len(CANONICAL_LANDMARK_INDICES), 2)
    for i, idx in enumerate(CANONICAL_LANDMARK_INDICES):
        assert np.array_equal(canonical[i], dense_pts[idx])


def test_extract_canonical_landmarks_invalid_input():
    """extract_canonical_landmarks raises ValueError when dense landmarks count is < 468."""
    with pytest.raises(ValueError):
        extract_canonical_landmarks(None)
    with pytest.raises(ValueError):
        extract_canonical_landmarks(np.zeros((100, 2), dtype=np.float32))


def test_compute_bounding_box_standard():
    """compute_bounding_box applies margin correctly around facial landmark bounds."""
    # Mock landmarks spanning x: [200, 400], y: [100, 300]
    landmarks = np.array([[200, 100], [400, 300]], dtype=np.float32)
    frame_shape = (480, 640)
    margin = 0.1  # width=200 -> dx=20; height=200 -> dy=20

    bbox = compute_bounding_box(landmarks, frame_shape=frame_shape, margin=margin)
    x_min, y_min, x_max, y_max = bbox

    assert x_min == 180
    assert y_min == 80
    assert x_max == 420
    assert y_max == 320


def test_compute_bounding_box_clamping_at_frame_boundaries():
    """compute_bounding_box clamps coordinates strictly within [0, W] and [0, H]."""
    # Landmarks near edge of 640x480 frame
    landmarks = np.array([[5, 5], [635, 475]], dtype=np.float32)
    frame_shape = (480, 640)
    margin = 0.2  # Would push past boundaries

    bbox = compute_bounding_box(landmarks, frame_shape=frame_shape, margin=margin)
    x_min, y_min, x_max, y_max = bbox

    assert x_min >= 0
    assert y_min >= 0
    assert x_max <= 640
    assert y_max <= 480


def test_face_oval_contour_closed_polygon(synthetic_face_image):
    """get_face_oval_contour returns a valid closed 36-point polygon with positive area."""
    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        landmarks, _ = tracker.process_frame(synthetic_face_image)

    contour = get_face_oval_contour(landmarks)
    assert contour.shape == (36, 2)
    area = cv2.contourArea(contour.astype(np.float32))
    assert area > 1000.0, "Face oval contour must have substantial positive area"


def test_canonical_delaunay_triangles_topology():
    """CANONICAL_DELAUNAY_TRIANGLES is valid, non-empty, and indexes canonical points."""
    triangles = np.array(CANONICAL_DELAUNAY_TRIANGLES)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert len(triangles) >= 100, f"Expected >= 100 triangles, got {len(triangles)}"

    # All vertex indices must be within canonical landmark bounds
    num_canonical = len(CANONICAL_LANDMARK_INDICES)
    assert np.all(triangles >= 0)
    assert np.all(triangles < num_canonical)


def test_canonical_delaunay_non_degenerate_on_synthetic_face(synthetic_face_image):
    """All Delaunay triangles on the synthetic face have non-zero positive area."""
    with FaceMeshTracker(static_image_mode=True, min_detection_confidence=0.1) as tracker:
        landmarks, _ = tracker.process_frame(synthetic_face_image)

    canonical = extract_canonical_landmarks(landmarks)
    triangles = np.array(CANONICAL_DELAUNAY_TRIANGLES)

    # Compute signed 2D triangle area: 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    v1 = canonical[triangles[:, 0], :2]
    v2 = canonical[triangles[:, 1], :2]
    v3 = canonical[triangles[:, 2], :2]

    areas = 0.5 * np.abs(
        (v2[:, 0] - v1[:, 0]) * (v3[:, 1] - v1[:, 1]) -
        (v3[:, 0] - v1[:, 0]) * (v2[:, 1] - v1[:, 1])
    )

    assert np.all(areas > 0.0), "Found degenerate (zero-area) Delaunay triangle"


def test_canonical_triangulation_stability(synthetic_face_image):
    """CanonicalLandmarks.get_static_triangles returns cached topology."""
    with FaceMeshTracker(static_mode=True, min_detection_confidence=0.1) as tracker:
        res = tracker.process(synthetic_face_image)
        triangles = CanonicalLandmarks.get_static_triangles(res.canonical_px)
        assert len(triangles) > 150
        # Subsequent calls must return cached topology
        assert CanonicalLandmarks.get_static_triangles() is triangles


def test_face_bounding_box_methods():
    """FaceBoundingBox supports tuple unpacking, indexing, iter, and slice."""
    bbox = FaceBoundingBox(x1=10, y1=20, x2=110, y2=170)
    assert bbox.width == 100
    assert bbox.height == 150
    assert bbox.area == 15000
    assert bbox.as_tuple() == (10, 20, 110, 170)
    assert len(bbox) == 4
    assert bbox[0] == 10 and bbox[1] == 20 and bbox[2] == 110 and bbox[3] == 170

    x1, y1, x2, y2 = bbox
    assert (x1, y1, x2, y2) == (10, 20, 110, 170)

    r_slice, c_slice = bbox.roi_slice
    assert r_slice == slice(20, 170)
    assert c_slice == slice(10, 110)


def test_face_mesh_tracker_reset_and_close(synthetic_face_image):
    """FaceMeshTracker reset() and close() methods."""
    tracker = FaceMeshTracker(static_mode=False)
    tracker.process(synthetic_face_image)
    assert tracker._prev_landmarks is not None

    tracker.reset()
    assert tracker._prev_landmarks is None

    tracker.close()
    assert tracker._mesh is None
    # Safe to call close() again
    tracker.close()


def test_face_mesh_tracker_grayscale_and_rgba(synthetic_face_image):
    """FaceMeshTracker accepts 1-channel grayscale and 4-channel RGBA images."""
    gray = cv2.cvtColor(synthetic_face_image, cv2.COLOR_BGR2GRAY)
    rgba = cv2.cvtColor(synthetic_face_image, cv2.COLOR_BGR2BGRA)

    with FaceMeshTracker(static_mode=True, min_detection_confidence=0.1) as tracker:
        res_gray = tracker.process(gray)
        assert res_gray.has_face is True

        res_rgba = tracker.process(rgba)
        assert res_rgba.has_face is True


def test_compute_delaunay_triangles_small_point_set():
    """compute_delaunay_triangles returns empty array when given < 3 points."""
    pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
    triangles = CanonicalLandmarks.compute_delaunay_triangles(pts)
    assert len(triangles) == 0


def test_get_face_oval_contour_from_canonical(synthetic_face_image):
    """get_face_oval_contour extracts contour when passed canonical 130 landmarks."""
    with FaceMeshTracker(static_mode=True, min_detection_confidence=0.1) as tracker:
        res = tracker.process(synthetic_face_image)

    contour = get_face_oval_contour(res.canonical_px)
    assert contour.shape == (36, 2)


def test_face_mesh_tracker_multi_face_selection(multi_face_image):
    """FaceMeshTracker selects primary (largest area) face when multi faces detected."""
    with FaceMeshTracker(static_mode=True, max_num_faces=4, min_detection_confidence=0.1) as tracker:
        res = tracker.process(multi_face_image)
        assert res.has_face is True
        assert res.bbox is not None
        # Primary face center is around x=200, y=240, bbox should enclose it
        assert res.bbox.x1 < 250


def test_face_mesh_tracker_float_dtype_inputs(synthetic_face_image):
    """FaceMeshTracker accepts float32 images in [0, 1] and [0, 255] without raising TypeError."""
    float_01 = (synthetic_face_image.astype(np.float32) / 255.0)
    float_255 = synthetic_face_image.astype(np.float64)

    with FaceMeshTracker(static_mode=True, min_detection_confidence=0.1) as tracker:
        res_01 = tracker.process(float_01)
        assert res_01.has_face is True

        res_255 = tracker.process(float_255)
        assert res_255.has_face is True


def test_extract_canonical_landmarks_python_list():
    """extract_canonical_landmarks accepts Python lists of 468 [x, y] coordinates."""
    mock_lms = [[float(i), float(i * 2)] for i in range(468)]
    canonical = extract_canonical_landmarks(mock_lms)
    assert isinstance(canonical, np.ndarray)
    assert canonical.shape == (130, 2)
    assert canonical[0, 0] == mock_lms[CanonicalLandmarks.CANONICAL_INDICES[0]][0]

