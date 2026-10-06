"""
deepfeik.core

Core Face Swap Engine: landmark tracking, canonical mesh topology,
piecewise affine warping, LAB color transfer, and feathered blending.
"""

from deepfeik.core.landmarks import (
    CanonicalLandmarks,
    CANONICAL_LANDMARK_INDICES,
    FACEMESH_FACE_OVAL,
    CANONICAL_DELAUNAY_TRIANGLES,
    extract_canonical_landmarks,
    compute_bounding_box,
    get_face_oval_contour,
)
from deepfeik.core.face_mesh import (
    FaceMeshTracker,
    FaceBoundingBox,
    FaceMeshResult,
)
from deepfeik.core.warper import (
    PiecewiseAffineWarper,
)
from deepfeik.core.color import (
    ReinhardColorMatcher,
)
from deepfeik.core.blender import (
    FeatheredAlphaBlender,
)
from deepfeik.core.engine import (
    FaceSwapEngine,
)

__all__ = [
    "CanonicalLandmarks",
    "CANONICAL_LANDMARK_INDICES",
    "FACEMESH_FACE_OVAL",
    "CANONICAL_DELAUNAY_TRIANGLES",
    "extract_canonical_landmarks",
    "compute_bounding_box",
    "get_face_oval_contour",
    "FaceMeshTracker",
    "FaceBoundingBox",
    "FaceMeshResult",
    "PiecewiseAffineWarper",
    "ReinhardColorMatcher",
    "FeatheredAlphaBlender",
    "FaceSwapEngine",
]
