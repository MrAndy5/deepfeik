"""
deepfeik.core.face_mesh

Real-time MediaPipe FaceMesh tracker supporting video tracking and static image modes,
dense 468/478 facial landmark extraction, bounding box calculation with margins,
and canonical subset extraction.
"""

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
except (AttributeError, ModuleNotFoundError):
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

from deepfeik.core.landmarks import CanonicalLandmarks


@dataclass(frozen=True)
class FaceBoundingBox:
    """Integer bounding box with margin and image boundary clipping."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def roi_slice(self) -> Tuple[slice, slice]:
        """Returns (row_slice, col_slice) for NumPy array indexing: frame[y1:y2, x1:x2]."""
        return (slice(self.y1, self.y2), slice(self.x1, self.x2))

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def __iter__(self) -> Iterator[int]:
        return iter((self.x1, self.y1, self.x2, self.y2))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> int:
        return (self.x1, self.y1, self.x2, self.y2)[idx]


@dataclass
class FaceMeshResult:
    """Structured detection and landmark result for a single processed frame."""
    has_face: bool
    landmarks_norm: Optional[np.ndarray] = None      # Shape (N, 2), normalized float32 in [0, 1]
    landmarks_px: Optional[np.ndarray] = None        # Shape (N, 2), pixel float32 [0..w, 0..h]
    landmarks_int: Optional[np.ndarray] = None       # Shape (N, 2), pixel int32 [0..w-1, 0..h-1]
    canonical_px: Optional[np.ndarray] = None        # Shape (130, 2), canonical landmark pixel float32
    bbox: Optional[FaceBoundingBox] = None           # Bounding box with margin
    face_oval: Optional[np.ndarray] = None           # Shape (36, 2), ordered face oval contour int32
    convex_hull: Optional[np.ndarray] = None         # Shape (K, 2), convex hull contour int32


class FaceMeshTracker:
    """MediaPipe FaceMesh wrapper supporting video tracking and static image modes.

    Attributes:
        static_mode: If False, uses previous-frame ROI tracking for high-FPS video.
                     If True, runs full-frame detection on each image.
        max_num_faces: Maximum number of faces to detect.
        refine_landmarks: If True, outputs 478 landmarks (including irises).
        bbox_margin: Margin ratio added to the landmark bounding box (default 0.15 = 15%).
        smooth_landmarks: If True and in video mode, applies EMA temporal smoothing.
    """

    def __init__(
        self,
        static_mode: bool = False,
        static_image_mode: Optional[bool] = None,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        bbox_margin: float = 0.15,
        smooth_landmarks: bool = True,
        smoothing_factor: float = 0.50,
    ):
        resolved_static = static_image_mode if static_image_mode is not None else static_mode
        self.static_mode = bool(resolved_static)
        self.max_num_faces = int(max_num_faces)
        self.refine_landmarks = bool(refine_landmarks)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.bbox_margin = float(bbox_margin)
        self.smooth_landmarks = bool(smooth_landmarks) and (not self.static_mode)
        self.smoothing_factor = float(smoothing_factor)

        self._prev_landmarks: Optional[np.ndarray] = None

        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=self.static_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def process(self, frame_bgr: Optional[np.ndarray]) -> FaceMeshResult:
        """Processes a BGR image/frame and extracts face landmarks and bounding box.

        Args:
            frame_bgr: BGR image as uint8 NumPy array.

        Returns:
            FaceMeshResult: Object containing detection flag, coordinates, bbox, and contours.
        """
        if frame_bgr is None or frame_bgr.size == 0 or self._mesh is None:
            self._prev_landmarks = None
            return FaceMeshResult(has_face=False)

        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            self._prev_landmarks = None
            return FaceMeshResult(has_face=False)

        # Defensive conversion: Ensure uint8 format for MediaPipe
        if np.issubdtype(frame_bgr.dtype, np.floating):
            if frame_bgr.max() <= 1.0:
                frame_bgr = np.clip(frame_bgr * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        elif frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        # 1. Color conversion: MediaPipe requires RGB input
        if len(frame_bgr.shape) == 2:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        elif frame_bgr.shape[2] == 4:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. Execute FaceMesh inference
        results = self._mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self._prev_landmarks = None
            return FaceMeshResult(has_face=False)

        # 3. Handle face selection
        face_landmarks_list = results.multi_face_landmarks
        if len(face_landmarks_list) == 1:
            selected_landmarks = face_landmarks_list[0]
        else:
            selected_landmarks = self._select_primary_face(face_landmarks_list)

        # 4. Extract normalized coordinates
        lms = selected_landmarks.landmark
        raw_norm = np.array([(lm.x, lm.y) for lm in lms], dtype=np.float32)

        # 5. Denormalize to pixel coordinates (float32)
        pixel_coords = np.empty_like(raw_norm)
        pixel_coords[:, 0] = raw_norm[:, 0] * w
        pixel_coords[:, 1] = raw_norm[:, 1] * h

        # 6. Apply temporal smoothing if enabled (video mode only)
        if self.smooth_landmarks and self._prev_landmarks is not None and self._prev_landmarks.shape == pixel_coords.shape:
            pixel_coords = (
                self.smoothing_factor * pixel_coords +
                (1.0 - self.smoothing_factor) * self._prev_landmarks
            )
        self._prev_landmarks = pixel_coords.copy()

        # 7. Clamped integer coordinates for rasterization and masks (NumPy 2.x safe)
        coords_int = np.clip(
            np.round(pixel_coords),
            [0, 0],
            [w - 1, h - 1]
        ).astype(np.int32)

        # 8. Bounding box computation with margin
        x_min, y_min = np.min(pixel_coords[:, 0]), np.min(pixel_coords[:, 1])
        x_max, y_max = np.max(pixel_coords[:, 0]), np.max(pixel_coords[:, 1])
        box_w = max(1.0, x_max - x_min)
        box_h = max(1.0, y_max - y_min)

        pad_x = self.bbox_margin * box_w
        pad_y = self.bbox_margin * box_h

        x1 = max(0, min(w, int(np.floor(x_min - pad_x))))
        y1 = max(0, min(h, int(np.floor(y_min - pad_y))))
        x2 = max(0, min(w, int(np.ceil(x_max + pad_x))))
        y2 = max(0, min(h, int(np.ceil(y_max + pad_y))))

        # Degenerate bbox check
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
            return FaceMeshResult(has_face=False)

        bbox = FaceBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        # 9. Canonical landmarks extraction (~130 key landmarks)
        canonical_px = CanonicalLandmarks.extract_canonical_landmarks(pixel_coords)

        # 10. Face Oval contour extraction (36 ordered perimeter points)
        face_oval = coords_int[list(CanonicalLandmarks.FACE_OVAL)]

        # 11. Convex Hull contour extraction
        convex_hull = cv2.convexHull(coords_int).reshape(-1, 2)

        return FaceMeshResult(
            has_face=True,
            landmarks_norm=raw_norm,
            landmarks_px=pixel_coords,
            landmarks_int=coords_int,
            canonical_px=canonical_px,
            bbox=bbox,
            face_oval=face_oval,
            convex_hull=convex_hull
        )

    def process_frame(
        self, frame_bgr: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[FaceBoundingBox]]:
        """Processes a frame and returns (landmarks_px, bbox) tuple for pipeline/test consumers.

        Args:
            frame_bgr: Input BGR frame.

        Returns:
            Tuple[Optional[np.ndarray], Optional[FaceBoundingBox]]:
                - (landmarks_px, bbox) if a face is detected.
                - (None, None) if no face is detected or frame is empty.
        """
        result = self.process(frame_bgr)
        if not result.has_face:
            return (None, None)
        return (result.landmarks_px, result.bbox)

    def _select_primary_face(self, face_landmarks_list: Sequence) -> Any:
        """Selects the face with the largest bounding box area from candidate faces."""
        max_area = -1.0
        best_face = face_landmarks_list[0]
        for face_lms in face_landmarks_list:
            xs = [lm.x for lm in face_lms.landmark]
            ys = [lm.y for lm in face_lms.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > max_area:
                max_area = area
                best_face = face_lms
        return best_face

    def reset(self) -> None:
        """Resets tracking history and temporal smoothing state."""
        self._prev_landmarks = None

    def close(self) -> None:
        """Closes the underlying MediaPipe FaceMesh model and releases native resources."""
        if hasattr(self, '_mesh') and self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
            self._mesh = None
        self._prev_landmarks = None

    def __enter__(self) -> "FaceMeshTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
