"""
deepfeik.core.engine

High-level FaceSwapEngine API coordinating landmark tracking,
piecewise affine warping, LAB color transfer, and feathered blending.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from deepfeik.core.face_mesh import FaceMeshTracker, FaceMeshResult, FaceBoundingBox
from deepfeik.core.warper import PiecewiseAffineWarper
from deepfeik.core.color import ReinhardColorMatcher
from deepfeik.core.blender import FeatheredAlphaBlender

# MediaPipe FaceMesh indices for inner lip contour (12 points).
# Used to cut a hole in the blend mask when the mouth is open so
# teeth / tongue / cavity show the live camera frame instead of
# smeared warped source pixels — this fixes the speaking artifact.
_LIPS_INNER_MP = (78, 80, 82, 13, 312, 310, 308, 318, 317, 14, 87, 88)

# MediaPipe index 13 = upper inner-lip midpoint, 14 = lower inner-lip midpoint.
_MOUTH_UPPER_IDX = 13
_MOUTH_LOWER_IDX = 14

# Pixel distance (after denormalization) between the two inner-lip midpoints
# required to classify the mouth as "open" and trigger the mask cutout.
_MOUTH_OPEN_PX_THRESHOLD = 6.0


class FaceSwapEngine:
    """Core Face Swap Engine implementing the deepfeik public engine contract.

    Zero GUI and zero hardware dependencies (Android & CI compatible).
    """

    def __init__(self, blend_mode: str = "feathered") -> None:
        """Initializes FaceSwapEngine with tracking, warping, color matching, and blending subsystems.

        Args:
            blend_mode: Blending method ('feathered' for fast real-time CPU or 'poisson').
        """
        self.blend_mode = blend_mode

        # Trackers: static for source image, video mode for live webcam stream
        self._static_tracker = FaceMeshTracker(
            static_mode=True,
            max_num_faces=4,
            min_detection_confidence=0.35,
        )
        self._video_tracker = FaceMeshTracker(
            static_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
            smooth_landmarks=True,
            smoothing_factor=0.80,
        )

        # Core CV modules
        self._warper = PiecewiseAffineWarper()
        self._color_matcher = ReinhardColorMatcher()
        self._blender = FeatheredAlphaBlender(erode_pixels=2, blur_kernel_size=17)

        # Active source face state
        self._source_face_loaded: bool = False
        self._source_image: Optional[np.ndarray] = None
        self._source_landmarks: Optional[np.ndarray] = None
        self._source_bbox: Optional[FaceBoundingBox] = None
        self._source_oval: Optional[np.ndarray] = None
        self._source_mask: Optional[np.ndarray] = None

    def set_source_face(self, image: Optional[np.ndarray]) -> Tuple[bool, str]:
        """Loads and processes source face from an image array.

        Detects face in source photo using FaceMeshTracker in static mode.
        If multiple faces are detected, automatically selects the primary face.
        Pre-computes and caches canonical landmarks, bounding box, and face mask.

        Args:
            image: Image array (BGR, BGRA, or Grayscale uint8).

        Returns:
            Tuple[bool, str]: (success, message). If success is False, message explains failure.
        """
        if image is None:
            return False, "Invalid image: None provided."

        if not isinstance(image, np.ndarray) or image.size == 0:
            return False, "Invalid image: Empty or non-numpy array."

        # Normalize channels to 3-channel BGR
        if len(image.shape) == 2:
            bgr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] == 3:
            bgr_img = image.copy()
        else:
            return False, f"Unsupported channel count: {image.shape[2]}"

        # Protect against ultra-high-resolution images (e.g. 4000x3000)
        h, w = bgr_img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1920:
            scale = 1920.0 / max_dim
            bgr_img = cv2.resize(
                bgr_img,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # Detect face in static image mode
        result: FaceMeshResult = self._static_tracker.process(bgr_img)
        if not result.has_face or result.canonical_px is None:
            return False, "No face detected in source image."

        # Successfully detected new face -> update cached state
        self._source_image = bgr_img
        self._source_landmarks = result.canonical_px
        self._source_bbox = result.bbox
        self._source_oval = result.face_oval

        # Pre-compute and cache face mask and color transfer statistics
        self._source_mask = self._blender.create_binary_mask(
            bgr_img.shape[:2], result.face_oval
        )
        self._warper.set_source(bgr_img, result.canonical_px)
        self._color_matcher.set_source_stats(bgr_img, self._source_mask)
        self._source_face_loaded = True

        return True, "Source face loaded successfully."

    def clear_source_face(self) -> None:
        """Clears active source face; process_frame reverts to passthrough."""
        self._source_face_loaded = False
        self._source_image = None
        self._source_landmarks = None
        self._source_bbox = None
        self._source_oval = None
        self._source_mask = None
        self._warper.clear_source()
        self._color_matcher.clear_source_stats()

    def is_source_loaded(self) -> bool:
        """Returns True if a valid source face is currently loaded."""
        return self._source_face_loaded

    def process_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """Processes a single BGR camera frame.

        If no source face or no face in frame, returns original frame unmodified.
        Otherwise returns seamlessly swapped BGR frame with correct mouth handling
        (inner mouth cavity shows the live camera when the mouth is open).

        Args:
            frame: Camera frame as BGR uint8 array.

        Returns:
            np.ndarray: Swapped BGR frame or original frame unaltered.
        """
        if frame is None or frame.size == 0:
            return frame

        # Passthrough if no source face loaded
        if not self._source_face_loaded:
            return frame

        # Detect face in live camera frame using video tracker
        res: FaceMeshResult = self._video_tracker.process(frame)
        if not res.has_face or res.canonical_px is None or res.face_oval is None:
            return frame

        target_landmarks = res.canonical_px
        target_oval = res.face_oval
        target_h, target_w = frame.shape[:2]

        # 1. Warp cached source face to target facial geometry
        warped = self._warper.warp_to_target(
            target_landmarks,
            (target_h, target_w),
        )

        # 2. Binary mask for the full face oval
        target_mask = self._blender.create_binary_mask(
            (target_h, target_w),
            target_oval,
        )

        # 3. Mouth-open cutout — fixes speaking artifact.
        #    When the mouth is open, the inner mouth region should show the
        #    live camera (teeth/tongue/cavity), NOT the smeared warped face.
        #    We detect this using full 468 MediaPipe landmarks and punch a
        #    hole in the blend mask at the inner-lip contour.
        if res.landmarks_px is not None and len(res.landmarks_px) >= 468:
            lms = res.landmarks_px  # shape (N, 2) float32 pixel coords

            upper_inner = lms[_MOUTH_UPPER_IDX]
            lower_inner = lms[_MOUTH_LOWER_IDX]
            mouth_open_px = float(np.linalg.norm(upper_inner - lower_inner))

            if mouth_open_px >= _MOUTH_OPEN_PX_THRESHOLD:
                # Build inner mouth contour from full-landmark indices
                inner_pts = np.array(
                    [lms[i] for i in _LIPS_INNER_MP], dtype=np.int32
                )
                # Scale the hole slightly larger than the inner contour so the
                # effect is visible even at the very edge of the lips.
                cx = float(inner_pts[:, 0].mean())
                cy = float(inner_pts[:, 1].mean())
                scale = 1.0 + min(0.4, mouth_open_px / 60.0)
                scaled_pts = np.array(
                    [(cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale)
                     for p in inner_pts],
                    dtype=np.int32,
                )
                cv2.fillPoly(target_mask, [scaled_pts.reshape(-1, 1, 2)], 0)

        # 4. LAB Reinhard color matching
        warped_matched = self._color_matcher.match(
            warped,
            frame,
            source_mask=target_mask,
            target_mask=target_mask,
            match_source_to_target=True,
        )

        # 5. Feathered alpha blending
        blended = self._blender.blend(
            warped_matched,
            frame,
            face_oval_points=target_oval,
            binary_mask=target_mask,
            mode=self.blend_mode,
        )

        return blended


    def reset(self) -> None:
        """Resets tracking state and temporal smoothing filter."""
        self._video_tracker.reset()

    def close(self) -> None:
        """Releases underlying MediaPipe tracker resources."""
        if hasattr(self, "_static_tracker") and self._static_tracker is not None:
            self._static_tracker.close()
        if hasattr(self, "_video_tracker") and self._video_tracker is not None:
            self._video_tracker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
