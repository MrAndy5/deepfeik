"""
deepfeik.core.blender

Feathered alpha blending using MediaPipe face oval contour.
Provides sub-1.5 ms real-time alpha compositing and optional Poisson seamless clone.
"""

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np

from deepfeik.core.landmarks import CanonicalLandmarks, FACEMESH_FACE_OVAL


class FeatheredAlphaBlender:
    """Seamless face blending engine using eroded and Gaussian-feathered contour masks."""

    def __init__(
        self,
        erode_pixels: int = 2,
        blur_kernel_size: int = 17,
        sigma: float = 0.0,
    ) -> None:
        """Initializes FeatheredAlphaBlender.

        Args:
            erode_pixels: Number of pixels to erode binary face mask before blurring.
            blur_kernel_size: Odd kernel size for Gaussian blur feathering (e.g. 15-21).
            sigma: Gaussian blur standard deviation (0 = auto based on kernel size).
        """
        self.erode_pixels = max(0, int(erode_pixels))
        # Ensure odd kernel size
        k = int(blur_kernel_size)
        self.blur_kernel_size = k if (k % 2 == 1) else (k + 1)
        self.sigma = float(sigma)

        # Pre-compute structuring element for erosion (fast 3x3 / 5x5)
        if self.erode_pixels > 0:
            k_size = 2 * self.erode_pixels + 1
            self._erode_kernel = np.ones((k_size, k_size), dtype=np.uint8)
        else:
            self._erode_kernel = None

    def create_binary_mask(
        self,
        shape: Tuple[int, ...],
        face_oval_points: np.ndarray,
    ) -> np.ndarray:
        """Generates a binary (0 or 255) mask from face oval points.

        Args:
            shape: (height, width) or (height, width, channels) of output mask.
            face_oval_points: Array of (K, 2) contour points.

        Returns:
            np.ndarray: uint8 binary mask of shape (height, width).
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if face_oval_points is None or len(face_oval_points) < 3:
            return mask

        pts = np.asarray(face_oval_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def create_feathered_mask(
        self,
        shape: Tuple[int, ...],
        face_oval_points: Optional[np.ndarray] = None,
        binary_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generates an eroded and Gaussian-blurred alpha mask.

        Args:
            shape: Canvas dimensions (height, width).
            face_oval_points: Contour coordinates (K, 2).
            binary_mask: Optional pre-existing binary mask. If provided, points are ignored.

        Returns:
            np.ndarray: float32 alpha mask of shape (height, width) with values in [0.0, 1.0].
        """
        h, w = shape[:2]
        if binary_mask is not None:
            mask = binary_mask.copy()
        elif face_oval_points is not None:
            mask = self.create_binary_mask(shape, face_oval_points)
        else:
            return np.zeros((h, w), dtype=np.float32)

        # 1. Erode boundary
        if self._erode_kernel is not None:
            mask = cv2.erode(mask, self._erode_kernel, iterations=1)

        # 2. Gaussian blur feathering
        if self.blur_kernel_size > 1:
            mask = cv2.GaussianBlur(
                mask,
                (self.blur_kernel_size, self.blur_kernel_size),
                self.sigma,
            )

        # Normalize to float32 [0.0, 1.0]
        alpha = mask.astype(np.float32) * (1.0 / 255.0)
        return alpha

    def blend(
        self,
        warped_face: np.ndarray,
        target_frame: np.ndarray,
        face_oval_points: Optional[np.ndarray] = None,
        alpha_mask: Optional[np.ndarray] = None,
        binary_mask: Optional[np.ndarray] = None,
        mode: str = "feathered",
    ) -> np.ndarray:
        """Composites warped face onto target camera frame.

        Args:
            warped_face: Warped face image (BGR uint8).
            target_frame: Target camera frame (BGR uint8).
            face_oval_points: Optional 36-point face oval contour.
            alpha_mask: Optional pre-computed float32 [0..1] mask.
            binary_mask: Optional binary mask for Poisson clone.
            mode: 'feathered' (default, < 1.5ms) or 'poisson' (seamlessClone, slow).

        Returns:
            np.ndarray: Blended BGR uint8 frame.
        """
        if warped_face is None or target_frame is None:
            return target_frame if target_frame is not None else warped_face

        h, w = target_frame.shape[:2]

        if mode.lower() in ("poisson", "seamless_clone"):
            return self._blend_poisson(
                warped_face, target_frame, face_oval_points, binary_mask
            )

        # Optimized bounding-box ROI path when face_oval_points provided
        if face_oval_points is not None and len(face_oval_points) >= 3 and alpha_mask is None:
            pts = np.asarray(face_oval_points, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            if rw <= 0 or rh <= 0:
                return target_frame

            pad = self.blur_kernel_size + self.erode_pixels + 5
            x1 = max(0, rx - pad)
            y1 = max(0, ry - pad)
            x2 = min(w, rx + rw + pad)
            y2 = min(h, ry + rh + pad)

            if x2 <= x1 or y2 <= y1:
                return target_frame

            roi_w = x2 - x1
            roi_h = y2 - y1

            # Local mask on ROI
            local_oval = pts - np.array([x1, y1], dtype=np.int32)
            roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [local_oval], 255)

            if self._erode_kernel is not None:
                roi_mask = cv2.erode(roi_mask, self._erode_kernel, iterations=1)

            if self.blur_kernel_size > 1:
                roi_mask = cv2.GaussianBlur(
                    roi_mask,
                    (self.blur_kernel_size, self.blur_kernel_size),
                    self.sigma,
                )

            alpha_roi = roi_mask.astype(np.float32) * (1.0 / 255.0)

            w_roi = warped_face[y1:y2, x1:x2]
            t_roi = target_frame[y1:y2, x1:x2]

            # cv2.blendLinear: ultra-fast AVX2 C++ SIMD (< 0.4 ms)
            blended_roi = cv2.blendLinear(w_roi, t_roi, alpha_roi, 1.0 - alpha_roi)

            out = target_frame.copy()
            out[y1:y2, x1:x2] = blended_roi
            return out

        # Full-frame fallback
        if alpha_mask is None:
            alpha = self.create_feathered_mask(
                (h, w),
                face_oval_points=face_oval_points,
                binary_mask=binary_mask,
            )
        else:
            alpha = alpha_mask

        # Ensure single channel float32 for cv2.blendLinear
        if len(alpha.shape) == 3:
            alpha = alpha[:, :, 0]
        alpha = alpha.astype(np.float32)

        return cv2.blendLinear(warped_face, target_frame, alpha, 1.0 - alpha)

    def _blend_poisson(
        self,
        warped: np.ndarray,
        target: np.ndarray,
        face_oval_points: Optional[np.ndarray],
        binary_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Applies cv2.seamlessClone (Poisson blending) with graceful fallback."""
        h, w = target.shape[:2]
        if binary_mask is None:
            if face_oval_points is not None:
                binary_mask = self.create_binary_mask((h, w), face_oval_points)
            else:
                return target

        # Check mask has content
        if cv2.countNonZero(binary_mask) == 0:
            return target

        # Find center of the mask
        m = cv2.moments(binary_mask)
        if m["m00"] > 0:
            cx = int(round(m["m10"] / m["m00"]))
            cy = int(round(m["m01"] / m["m00"]))
        else:
            cx, cy = w // 2, h // 2

        # Clamp center away from edges to prevent cv2.seamlessClone crash
        cx = max(10, min(w - 10, cx))
        cy = max(10, min(h - 10, cy))

        try:
            return cv2.seamlessClone(
                warped,
                target,
                binary_mask,
                (cx, cy),
                cv2.NORMAL_CLONE,
            )
        except Exception:
            # Fallback to feathered blend if Poisson fails
            return self.blend(warped, target, face_oval_points=face_oval_points, mode="feathered")
