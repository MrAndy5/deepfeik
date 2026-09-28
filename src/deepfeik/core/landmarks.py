"""
deepfeik.core.landmarks

Canonical landmark topology, anatomical groupings, Delaunay mesh definitions,
and bounding box geometry for 2D facial landmark warping.
"""

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np
from scipy.spatial import Delaunay


class CanonicalLandmarks:
    """Canonical face landmark definitions, groupings, and mesh topology."""

    # 1. Face Oval: 36 points tracing the continuous outer perimeter
    FACE_OVAL: Tuple[int, ...] = (
        10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58,
        172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
        365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332,
        297, 338
    )

    # 2. Eyebrows: 14 points (7 per brow)
    RIGHT_EYEBROW: Tuple[int, ...] = (70, 63, 105, 66, 107, 55, 65)
    LEFT_EYEBROW: Tuple[int, ...] = (336, 296, 334, 293, 300, 285, 295)

    # 3. Eyes: 20 points (10 per eye, perimeter + corners)
    RIGHT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144, 7, 163, 145, 154)
    LEFT_EYE: Tuple[int, ...] = (362, 385, 387, 263, 373, 380, 382, 381, 374, 390)

    # 4. Nose: 12 points (bridge, ridge, tip, nostrils)
    NOSE: Tuple[int, ...] = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327)

    # 5. Lips: 24 points (12 outer contour, 12 inner contour)
    LIPS_OUTER: Tuple[int, ...] = (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91)
    LIPS_INNER: Tuple[int, ...] = (78, 80, 82, 13, 312, 310, 308, 318, 317, 14, 87, 88)
    LIPS: Tuple[int, ...] = LIPS_OUTER + LIPS_INNER

    # 6. Cheeks, Forehead & Interior Anchors: 24 points to stabilize Delaunay triangles
    CHEEKS_FOREHEAD: Tuple[int, ...] = (
        # Forehead & Glabella
        9, 8, 151, 108, 337,
        # Right Cheek / Zygomatic & Bucca
        116, 123, 147, 213, 137, 205, 50, 192,
        # Left Cheek / Zygomatic & Bucca
        345, 352, 376, 433, 366, 425, 280, 416,
        # Chin Interior
        175, 199, 200
    )

    # The canonical ~130 key landmark subset (sorted for stable index alignment)
    CANONICAL_INDICES: np.ndarray = np.array(
        sorted(list(set(
            FACE_OVAL + RIGHT_EYEBROW + LEFT_EYEBROW +
            RIGHT_EYE + LEFT_EYE + NOSE + LIPS + CHEEKS_FOREHEAD
        ))),
        dtype=np.int32
    )

    # Cached static Delaunay topology
    _STATIC_TRIANGLES: Optional[np.ndarray] = None

    @classmethod
    def get_canonical_indices(cls) -> np.ndarray:
        """Returns the sorted array of canonical 130 landmark indices."""
        return cls.CANONICAL_INDICES

    @classmethod
    def extract_canonical_landmarks(cls, dense_landmarks: np.ndarray) -> np.ndarray:
        """Subsamples 468/478 dense landmarks to the canonical 130 landmarks.

        Args:
            dense_landmarks: np.ndarray of shape (N, 2) or (N, 3), where N >= 468.

        Returns:
            np.ndarray of shape (130, 2) or (130, 3).
        """
        if dense_landmarks is None or len(dense_landmarks) < 468:
            raise ValueError(
                f"Expected at least 468 landmarks, got {0 if dense_landmarks is None else len(dense_landmarks)}"
            )
        dense_landmarks = np.asarray(dense_landmarks)
        return dense_landmarks[cls.CANONICAL_INDICES]

    @classmethod
    def compute_delaunay_triangles(
        cls,
        landmarks_2d: np.ndarray,
        filter_inside_oval: bool = True,
        oval_indices: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Computes Delaunay triangulation simplices for the given 2D points.

        Args:
            landmarks_2d: np.ndarray of shape (K, 2) containing 2D coordinates.
            filter_inside_oval: If True, discards triangles whose centroid lies outside the face oval.
            oval_indices: Optional list of indices within landmarks_2d that form the face oval polygon.

        Returns:
            np.ndarray of shape (M, 3) of integer triangle vertex indices into landmarks_2d.
        """
        if len(landmarks_2d) < 3:
            return np.empty((0, 3), dtype=np.int32)

        tri = Delaunay(landmarks_2d[:, :2])
        simplices = tri.simplices.astype(np.int32)

        if not filter_inside_oval or oval_indices is None or len(oval_indices) < 3:
            return simplices

        oval_poly = np.array([landmarks_2d[i, :2] for i in oval_indices], dtype=np.float32)
        valid = []
        for s in simplices:
            centroid = tuple(map(float, np.mean(landmarks_2d[s, :2], axis=0)))
            if cv2.pointPolygonTest(oval_poly, centroid, False) >= 0:
                valid.append(s)

        return np.array(valid, dtype=np.int32) if valid else simplices

    @classmethod
    def get_static_triangles(cls, canonical_landmarks_sample: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the pre-computed static Delaunay triangulation topology.
        If not yet initialized and sample landmarks are provided, computes and caches it.
        """
        if cls._STATIC_TRIANGLES is not None and canonical_landmarks_sample is None:
            return cls._STATIC_TRIANGLES

        if canonical_landmarks_sample is not None:
            idx_map = {idx: i for i, idx in enumerate(cls.CANONICAL_INDICES)}
            local_oval = [idx_map[idx] for idx in cls.FACE_OVAL if idx in idx_map]
            computed = cls.compute_delaunay_triangles(
                canonical_landmarks_sample,
                filter_inside_oval=True,
                oval_indices=local_oval
            )
            if cls._STATIC_TRIANGLES is None or len(cls._STATIC_TRIANGLES) == 0:
                cls._STATIC_TRIANGLES = computed
            return cls._STATIC_TRIANGLES

        if cls._STATIC_TRIANGLES is not None:
            return cls._STATIC_TRIANGLES

        raise RuntimeError("Static triangles not initialized; provide sample canonical landmarks once.")


# Canonical topological triangles (221 valid facial triangles within face oval)
CANONICAL_DELAUNAY_TRIANGLES: np.ndarray = np.array(
    (
        (71, 72, 114), (68, 22, 65), (71, 53, 72), (46, 35, 52), (35, 44, 52),
        (43, 75, 45), (43, 44, 75), (39, 19, 38), (128, 126, 127), (113, 128, 127),
        (80, 113, 127), (113, 107, 128), (66, 114, 57), (66, 71, 114), (53, 66, 57),
        (66, 53, 71), (72, 125, 114), (93, 125, 72), (94, 93, 11), (21, 68, 65),
        (74, 46, 52), (74, 21, 46), (21, 74, 68), (31, 12, 72), (93, 12, 11),
        (12, 93, 72), (53, 67, 72), (67, 31, 72), (41, 42, 9), (56, 41, 9),
        (68, 73, 22), (73, 74, 52), (74, 73, 68), (37, 73, 51), (49, 35, 75),
        (44, 49, 75), (49, 44, 35), (19, 23, 14), (39, 23, 19), (107, 83, 128),
        (126, 83, 124), (83, 126, 128), (98, 78, 2), (105, 129, 110), (105, 113, 80),
        (113, 97, 107), (97, 105, 110), (105, 97, 113), (122, 89, 76), (89, 81, 76),
        (129, 104, 106), (104, 105, 80), (105, 104, 129), (69, 59, 47), (111, 80, 127),
        (111, 123, 80), (30, 10, 11), (10, 94, 11), (94, 10, 92), (115, 125, 93),
        (84, 126, 124), (126, 84, 127), (26, 42, 41), (26, 39, 38), (56, 40, 41),
        (24, 40, 20), (43, 18, 44), (44, 18, 52), (18, 73, 52), (27, 62, 14),
        (23, 27, 14), (62, 27, 45), (60, 24, 20), (47, 60, 20), (59, 60, 47),
        (99, 100, 88), (123, 77, 80), (77, 104, 80), (104, 77, 106), (77, 122, 106),
        (77, 89, 122), (59, 4, 37), (69, 4, 59), (4, 3, 37), (3, 4, 98),
        (98, 112, 127), (112, 111, 127), (0, 78, 92), (10, 0, 92), (78, 0, 2),
        (0, 10, 30), (32, 30, 11), (32, 33, 30), (33, 32, 31), (12, 32, 11),
        (32, 12, 31), (33, 29, 30), (95, 94, 92), (94, 95, 93), (109, 84, 124),
        (25, 24, 39), (26, 25, 39), (25, 26, 41), (40, 25, 41), (25, 40, 24),
        (73, 50, 51), (18, 50, 73), (63, 50, 18), (50, 60, 51), (15, 43, 45),
        (27, 15, 45), (15, 27, 23), (15, 23, 39), (60, 58, 51), (58, 60, 59),
        (58, 37, 51), (58, 59, 37), (89, 85, 81), (81, 85, 99), (85, 100, 99),
        (77, 85, 89), (100, 87, 88), (1, 3, 98), (3, 1, 37), (118, 4, 69),
        (0, 16, 2), (16, 0, 30), (16, 37, 2), (29, 16, 30), (28, 29, 33),
        (17, 73, 37), (16, 17, 37), (17, 16, 29), (73, 17, 22), (17, 28, 22),
        (28, 17, 29), (67, 54, 31), (28, 34, 22), (34, 28, 33), (34, 33, 31),
        (54, 34, 31), (34, 54, 55), (91, 95, 92), (78, 91, 92), (96, 109, 116),
        (109, 96, 84), (115, 96, 116), (96, 115, 93), (95, 96, 93), (15, 6, 43),
        (6, 18, 43), (6, 63, 18), (24, 61, 39), (61, 15, 39), (60, 61, 24),
        (61, 6, 15), (6, 61, 63), (50, 61, 60), (61, 50, 63), (102, 103, 88),
        (87, 102, 88), (103, 102, 9), (102, 56, 9), (13, 1, 98), (1, 13, 37),
        (4, 117, 98), (118, 117, 4), (117, 112, 98), (117, 120, 112), (120, 117, 118),
        (22, 48, 65), (34, 48, 22), (48, 34, 55), (91, 90, 95), (96, 90, 84),
        (90, 96, 95), (79, 98, 127), (84, 79, 127), (79, 78, 98), (79, 91, 78),
        (90, 79, 84), (79, 90, 91), (112, 121, 111), (120, 121, 112), (111, 121, 123),
        (121, 77, 123), (85, 121, 100), (121, 85, 77), (36, 98, 2), (36, 13, 98),
        (37, 36, 2), (13, 36, 37), (86, 121, 120), (86, 87, 100), (121, 86, 100),
        (119, 118, 69), (108, 119, 69), (119, 120, 118), (119, 108, 120), (70, 108, 69),
        (108, 70, 5), (70, 69, 47), (5, 70, 47), (108, 82, 120), (82, 86, 120),
        (64, 108, 5), (64, 82, 108), (64, 47, 20), (64, 5, 47), (40, 8, 20),
        (8, 40, 56), (86, 101, 87), (82, 101, 86), (101, 102, 87), (8, 101, 82),
        (102, 101, 56), (101, 8, 56), (64, 7, 82), (7, 8, 82), (7, 64, 20),
        (8, 7, 20)
    ),
    dtype=np.int32
)

# Initialize static Delaunay topology cache
CanonicalLandmarks._STATIC_TRIANGLES = CANONICAL_DELAUNAY_TRIANGLES

# Module-level aliases and helper functions
CANONICAL_LANDMARK_INDICES: np.ndarray = CanonicalLandmarks.CANONICAL_INDICES
FACEMESH_FACE_OVAL: Tuple[int, ...] = CanonicalLandmarks.FACE_OVAL


def extract_canonical_landmarks(dense_landmarks: np.ndarray) -> np.ndarray:
    """Extracts canonical 130 landmarks from dense landmarks."""
    return CanonicalLandmarks.extract_canonical_landmarks(dense_landmarks)


def compute_bounding_box(
    landmarks: np.ndarray,
    frame_shape: Optional[Tuple[int, int]] = None,
    margin: float = 0.15,
) -> Tuple[int, int, int, int]:
    """
    Computes integer bounding box (x_min, y_min, x_max, y_max) with margin around landmarks,
    clamped to frame boundaries if frame_shape is provided.

    Args:
        landmarks: (N, 2) or (N, 3) array of landmark coordinates.
        frame_shape: Optional (height, width) tuple of the frame.
        margin: Ratio of width/height to add as padding on each side.

    Returns:
        Tuple[int, int, int, int]: (x_min, y_min, x_max, y_max)
    """
    pts = np.asarray(landmarks)
    x_min, y_min = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x_max, y_max = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    pad_x = w * margin
    pad_y = h * margin

    res_x_min = int(round(x_min - pad_x))
    res_y_min = int(round(y_min - pad_y))
    res_x_max = int(round(x_max + pad_x))
    res_y_max = int(round(y_max + pad_y))

    if frame_shape is not None:
        h_frame, w_frame = frame_shape[:2]
        res_x_min = max(0, min(w_frame, res_x_min))
        res_y_min = max(0, min(h_frame, res_y_min))
        res_x_max = max(0, min(w_frame, res_x_max))
        res_y_max = max(0, min(h_frame, res_y_max))

    return (res_x_min, res_y_min, res_x_max, res_y_max)


def get_face_oval_contour(landmarks: np.ndarray) -> np.ndarray:
    """
    Extracts the ordered 36-vertex face oval contour from facial landmarks.

    Args:
        landmarks: Dense landmarks (>=468 points) or canonical landmarks (130 points).

    Returns:
        np.ndarray of shape (36, 2) containing contour coordinates.
    """
    pts = np.asarray(landmarks)
    if len(pts) >= 468:
        return pts[list(CanonicalLandmarks.FACE_OVAL), :2]
    elif len(pts) == len(CanonicalLandmarks.CANONICAL_INDICES):
        idx_map = {idx: i for i, idx in enumerate(CanonicalLandmarks.CANONICAL_INDICES)}
        local_indices = [idx_map[idx] for idx in CanonicalLandmarks.FACE_OVAL if idx in idx_map]
        return pts[local_indices, :2]
    else:
        raise ValueError(
            f"Expected at least 468 dense or 130 canonical landmarks, got {len(pts)}"
        )
