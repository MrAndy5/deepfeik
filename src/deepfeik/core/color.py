"""
deepfeik.core.color

Reinhard Mean-Standard Deviation color transfer in LAB color space.
Matches skin tone and lighting disparity between source and target faces in < 3 ms.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class ReinhardColorMatcher:
    """Color transfer matching skin tone and illumination statistics in LAB color space."""

    def __init__(self, eps: float = 1e-4) -> None:
        """Initializes ReinhardColorMatcher.

        Args:
            eps: Epsilon to prevent division by zero when standard deviation is very small.
        """
        self.eps = eps
        self._cached_source_mean: Optional[np.ndarray] = None
        self._cached_source_std: Optional[np.ndarray] = None

    def set_source_stats(
        self,
        source_bgr: np.ndarray,
        source_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Pre-computes and caches LAB mean and standard deviation for the source face.

        Args:
            source_bgr: Source image (BGR uint8).
            source_mask: Optional binary mask (uint8) defining the face region.
        """
        mean, std = self.compute_lab_stats(source_bgr, source_mask)
        self._cached_source_mean = mean
        self._cached_source_std = std

    def clear_source_stats(self) -> None:
        """Clears cached source face color statistics."""
        self._cached_source_mean = None
        self._cached_source_std = None

    def compute_lab_stats(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes mean and standard deviation of L, a, b channels.

        Args:
            image_bgr: BGR image as uint8 array.
            mask: Optional single-channel uint8 mask.

        Returns:
            Tuple of (mean: np.ndarray, std: np.ndarray), each of shape (3,).
        """
        if image_bgr is None or image_bgr.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Ensure 3-channel BGR
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        valid_mask = None
        if mask is not None and mask.size > 0:
            if mask.shape[:2] == lab.shape[:2] and cv2.countNonZero(mask) > 0:
                valid_mask = mask

        mean_val, std_val = cv2.meanStdDev(lab, mask=valid_mask)
        mean_arr = mean_val.flatten().astype(np.float32)
        std_arr = np.maximum(std_val.flatten().astype(np.float32), self.eps)

        return mean_arr, std_arr

    def match(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        source_mask: Optional[np.ndarray] = None,
        target_mask: Optional[np.ndarray] = None,
        match_source_to_target: bool = True,
    ) -> np.ndarray:
        """Transfers color statistics between source and target images in LAB space.

        Args:
            source_bgr: Source image (BGR uint8) — typically the warped source face.
            target_bgr: Target image (BGR uint8) — the live camera frame or reference.
            source_mask: Optional uint8 mask on source image.
            target_mask: Optional uint8 mask on target image.
            match_source_to_target: If True (default), adapts source_bgr to target_bgr tone.
                                    If False, adapts target_bgr to source_bgr tone.

        Returns:
            np.ndarray: Color-adjusted image in BGR format.
        """
        if source_bgr is None or source_bgr.size == 0:
            return source_bgr

        if match_source_to_target:
            input_img = source_bgr
            input_mask = source_mask
            ref_img = target_bgr
            ref_mask = target_mask
            cached_mean = self._cached_source_mean
            cached_std = self._cached_source_std
        else:
            input_img = target_bgr
            input_mask = target_mask
            ref_img = source_bgr
            ref_mask = source_mask
            cached_mean = None
            cached_std = None

        # Convert input to LAB
        if len(input_img.shape) == 2:
            input_bgr = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        elif input_img.shape[2] == 4:
            input_bgr = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
        else:
            input_bgr = input_img

        input_lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB)

        # Compute or retrieve statistics
        if cached_mean is not None and cached_std is not None and input_mask is None:
            s_mean, s_std = cached_mean, cached_std
        else:
            s_mean, s_std = self.compute_lab_stats(input_bgr, input_mask)

        r_mean, r_std = self.compute_lab_stats(ref_img, ref_mask)

        # Compute scaling and shift
        scale = r_std / s_std
        shift = r_mean - s_mean * scale

        # Apply transformation
        lab_float = input_lab.astype(np.float32)
        out_lab = lab_float * scale + shift
        out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)

        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
        return out_bgr
