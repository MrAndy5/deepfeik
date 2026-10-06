"""
deepfeik.core.color

Reinhard Mean-Standard Deviation color transfer in LAB color space.
Matches skin tone and lighting disparity between source and target faces in < 3 ms.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


_LUT_X = np.arange(256, dtype=np.float32)[:, None]


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
        self._cached_target_mean: Optional[np.ndarray] = None
        self._cached_target_std: Optional[np.ndarray] = None

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

    def set_target_stats(
        self,
        target_bgr: np.ndarray,
        target_mask: Optional[np.ndarray] = None,
        smooth: bool = False,
        alpha: float = 0.3,
    ) -> None:
        """Pre-computes and caches LAB mean and standard deviation for the target face.

        Supports optional Exponential Moving Average (EMA) smoothing to prevent
        frame-to-frame illumination jitter across live webcam frames.

        Args:
            target_bgr: Target image or frame (BGR uint8).
            target_mask: Optional binary mask (uint8) defining the target face region.
            smooth: If True and previous target stats exist, applies EMA smoothing.
            alpha: Smoothing factor for current frame [0.0, 1.0]. Lower values yield
                   smoother transitions across frames.
        """
        if target_bgr is None or target_bgr.size == 0:
            return

        mean, std = self.compute_lab_stats(target_bgr, target_mask)

        if smooth and self._cached_target_mean is not None and self._cached_target_std is not None:
            self._cached_target_mean = (
                alpha * mean + (1.0 - alpha) * self._cached_target_mean
            ).astype(np.float32)
            self._cached_target_std = (
                alpha * std + (1.0 - alpha) * self._cached_target_std
            ).astype(np.float32)
        else:
            self._cached_target_mean = mean
            self._cached_target_std = std

    def clear_target_stats(self) -> None:
        """Clears cached target face color statistics."""
        self._cached_target_mean = None
        self._cached_target_std = None

    def compute_lab_stats(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes mean and standard deviation of L, a, b channels.

        Optimized to use bounding-box ROI cropping when a mask is provided,
        avoiding full-frame color conversions and statistics passes.

        Args:
            image_bgr: BGR image as uint8 array.
            mask: Optional single-channel uint8 mask.

        Returns:
            Tuple of (mean: np.ndarray, std: np.ndarray), each of shape (3,).
        """
        if image_bgr is None or image_bgr.size == 0:
            return (
                np.array([128.0, 128.0, 128.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )

        # Optimize with ROI if valid mask is provided
        use_roi = False
        valid_mask = None
        if mask is not None and mask.size > 0 and mask.shape[:2] == image_bgr.shape[:2]:
            rx, ry, rw, rh = cv2.boundingRect(mask)
            if rw > 0 and rh > 0:
                use_roi = True
                image_bgr = image_bgr[ry : ry + rh, rx : rx + rw]
                valid_mask = mask[ry : ry + rh, rx : rx + rw]

        # Ensure 3-channel BGR
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        if not use_roi and mask is not None and mask.size > 0:
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
        use_cached_source: bool = True,
        use_cached_target: bool = True,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Transfers color statistics between source and target images in LAB space.

        Args:
            source_bgr: Source image (BGR uint8) — typically the warped source face.
            target_bgr: Target image (BGR uint8) — the live camera frame or reference.
            source_mask: Optional uint8 mask on source image.
            target_mask: Optional uint8 mask on target image.
            match_source_to_target: If True (default), adapts source_bgr to target_bgr tone.
                                    If False, adapts target_bgr to source_bgr tone.
            use_cached_source: If True, reuses pre-cached source LAB statistics if available.
            use_cached_target: If True, reuses pre-cached target LAB statistics if available.
            roi_box: Optional (x, y, w, h) bounding box to restrict color transform to ROI.

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
            cached_input_mean = self._cached_source_mean if use_cached_source else None
            cached_input_std = self._cached_source_std if use_cached_source else None
            cached_ref_mean = self._cached_target_mean if use_cached_target else None
            cached_ref_std = self._cached_target_std if use_cached_target else None
        else:
            input_img = target_bgr
            input_mask = target_mask
            ref_img = source_bgr
            ref_mask = source_mask
            cached_input_mean = self._cached_target_mean if use_cached_target else None
            cached_input_std = self._cached_target_std if use_cached_target else None
            cached_ref_mean = self._cached_source_mean if use_cached_source else None
            cached_ref_std = self._cached_source_std if use_cached_source else None

        # Retrieve or compute statistics
        if cached_input_mean is not None and cached_input_std is not None:
            s_mean, s_std = cached_input_mean, cached_input_std
        else:
            s_mean, s_std = self.compute_lab_stats(input_img, input_mask)

        if cached_ref_mean is not None and cached_ref_std is not None:
            r_mean, r_std = cached_ref_mean, cached_ref_std
        else:
            r_mean, r_std = self.compute_lab_stats(ref_img, ref_mask)

        # Compute scaling and shift for LAB channels
        scale = r_std / s_std
        shift = r_mean - s_mean * scale

        # Precompute vectorized 256-entry Look-Up Table for fast SIMD transformation
        lut = np.clip(
            _LUT_X * scale + shift, 0.0, 255.0
        ).astype(np.uint8).reshape(1, 256, 3)

        # Normalize input to 3-channel BGR
        if len(input_img.shape) == 2:
            input_bgr = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        elif input_img.shape[2] == 4:
            input_bgr = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
        else:
            input_bgr = input_img

        # Determine ROI for bounded transformation
        if roi_box is None:
            if input_mask is not None and input_mask.shape[:2] == input_bgr.shape[:2]:
                bx, by, bw, bh = cv2.boundingRect(input_mask)
                if bw > 0 and bh > 0:
                    roi_box = (bx, by, bw, bh)
            elif ref_mask is not None and ref_mask.shape[:2] == input_bgr.shape[:2]:
                bx, by, bw, bh = cv2.boundingRect(ref_mask)
                if bw > 0 and bh > 0:
                    roi_box = (bx, by, bw, bh)

        if roi_box is not None:
            rx, ry, rw, rh = roi_box
            ih, iw = input_bgr.shape[:2]
            rx = max(0, min(rx, iw - 1))
            ry = max(0, min(ry, ih - 1))
            rw = max(0, min(rw, iw - rx))
            rh = max(0, min(rh, ih - ry))
            if rw > 0 and rh > 0:
                if rx == 0 and ry == 0 and rw == iw and rh == ih:
                    input_lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB)
                    out_lab = cv2.LUT(input_lab, lut)
                    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
                else:
                    roi = input_bgr[ry : ry + rh, rx : rx + rw]
                    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                    out_roi_lab = cv2.LUT(roi_lab, lut)
                    out_roi_bgr = cv2.cvtColor(out_roi_lab, cv2.COLOR_LAB2BGR)
                    out_bgr = input_bgr.copy()
                    out_bgr[ry : ry + rh, rx : rx + rw] = out_roi_bgr
                    return out_bgr

        # Full image fallback
        input_lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB)
        out_lab = cv2.LUT(input_lab, lut)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
        return out_bgr
