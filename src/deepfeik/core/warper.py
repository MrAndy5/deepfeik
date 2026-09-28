"""
deepfeik.core.warper

Piecewise affine warping using canonical Delaunay triangulation.
Optimized for low CPU latency (< 10 ms) on face regions.
"""

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np

from deepfeik.core.landmarks import CANONICAL_DELAUNAY_TRIANGLES, CanonicalLandmarks


class PiecewiseAffineWarper:
    """Warp a source face into a target facial geometry using piecewise affine

    transforms on canonical Delaunay simplices.
    """

    def __init__(self, triangles: Optional[np.ndarray] = None) -> None:
        """Initializes warper with Delaunay triangle topology.

        Args:
            triangles: Optional (M, 3) integer array of triangle vertex indices.
                       Defaults to CANONICAL_DELAUNAY_TRIANGLES (221 triangles).
        """
        if triangles is not None:
            self._triangles = np.asarray(triangles, dtype=np.int32)
        else:
            self._triangles = CANONICAL_DELAUNAY_TRIANGLES

        self._mask_buf = np.zeros((300, 300), dtype=np.uint8)
        self._cached_source_patches: Optional[list] = None
        self._cached_src_id: Optional[int] = None

    @property
    def triangles(self) -> np.ndarray:
        """Returns the triangle indices array of shape (M, 3)."""
        return self._triangles

    def set_source(self, src_img: np.ndarray, src_landmarks: np.ndarray) -> None:
        """Pre-computes and caches source triangle patches and local coordinates.

        Enables sub-10ms CPU latency during live camera streaming.

        Args:
            src_img: Source face BGR image (uint8).
            src_landmarks: Canonical 130 landmarks (130, 2) or dense (>=468, 2).
        """
        if src_img is None or src_landmarks is None:
            self.clear_source()
            return

        src_pts = np.asarray(src_landmarks, dtype=np.float32)
        if len(src_pts) >= 468:
            src_pts = CanonicalLandmarks.extract_canonical_landmarks(src_pts)[:, :2]

        h, w = src_img.shape[:2]
        cached_patches = []

        for tri in self._triangles:
            s_tri = src_pts[tri]  # shape (3, 2)

            sx, sy, sw, sh = cv2.boundingRect(np.int32([s_tri]))
            if sw <= 0 or sh <= 0:
                continue

            # Bounds clamping
            sx1 = max(0, min(w, sx))
            sy1 = max(0, min(h, sy))
            sx2 = max(0, min(w, sx + sw))
            sy2 = max(0, min(h, sy + sh))

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            raw_patch = src_img[sy1:sy2, sx1:sx2]
            pw, ph = sx2 - sx1, sy2 - sy1

            # If patch touches outer boundary and was clamped, pad to expected (sh, sw)
            if pw != sw or ph != sh:
                s_patch = np.zeros((sh, sw, src_img.shape[2]), dtype=src_img.dtype)
                off_x = sx1 - sx
                off_y = sy1 - sy
                s_patch[off_y : off_y + ph, off_x : off_x + pw] = raw_patch
            else:
                s_patch = raw_patch

            s_off = (s_tri - np.array([sx, sy], dtype=np.float32)).astype(np.float32)
            cached_patches.append((s_patch, s_off, tri))

        self._cached_source_patches = cached_patches
        self._cached_src_id = id(src_img)

    def clear_source(self) -> None:
        """Clears cached source face data."""
        self._cached_source_patches = None
        self._cached_src_id = None

    def has_cached_source(self) -> bool:
        """Returns True if source patches are cached."""
        return self._cached_source_patches is not None

    def warp_to_target(
        self,
        dst_landmarks: np.ndarray,
        dst_shape: Tuple[int, ...],
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Warps the pre-cached source face onto the target geometry.

        Args:
            dst_landmarks: Canonical 130 landmarks (130, 2) or dense (>=468, 2).
            dst_shape: Target frame shape (height, width) or (height, width, channels).
            out: Optional destination array to composite onto. If None, allocates zeros.

        Returns:
            np.ndarray: Warped image of shape (height, width, 3).
        """
        if self._cached_source_patches is None:
            raise RuntimeError("No cached source face; call set_source() before warp_to_target().")

        dst_h, dst_w = dst_shape[:2]
        if out is None:
            dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        else:
            dst = out

        dst_pts = np.asarray(dst_landmarks, dtype=np.float32)
        if len(dst_pts) >= 468:
            dst_pts = CanonicalLandmarks.extract_canonical_landmarks(dst_pts)[:, :2]

        mask_buf = self._mask_buf

        for s_patch, s_off, tri in self._cached_source_patches:
            d_tri = dst_pts[tri]  # shape (3, 2)

            rx, ry, rw, rh = cv2.boundingRect(np.int32([d_tri]))
            if rw <= 0 or rh <= 0:
                continue

            # Check if triangle lies entirely outside canvas
            if rx + rw <= 0 or ry + rh <= 0 or rx >= dst_w or ry >= dst_h:
                continue

            d_off = (d_tri - np.array([rx, ry], dtype=np.float32)).astype(np.float32)

            # Compute affine transform
            try:
                warp_mat = cv2.getAffineTransform(s_off, d_off)
            except Exception:
                continue

            # Warp patch
            warped_patch = cv2.warpAffine(
                s_patch,
                warp_mat,
                (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Ensure mask buffer is large enough
            if rh > mask_buf.shape[0] or rw > mask_buf.shape[1]:
                new_h = max(rh + 50, mask_buf.shape[0] * 2)
                new_w = max(rw + 50, mask_buf.shape[1] * 2)
                mask_buf = np.zeros((new_h, new_w), dtype=np.uint8)
                self._mask_buf = mask_buf

            m_view = mask_buf[:rh, :rw]
            m_view.fill(0)
            cv2.fillConvexPoly(m_view, np.int32(d_off), 255)

            # Destination clipping
            x1 = max(0, rx)
            y1 = max(0, ry)
            x2 = min(dst_w, rx + rw)
            y2 = min(dst_h, ry + rh)

            if x2 <= x1 or y2 <= y1:
                continue

            # Patch sub-rectangle corresponding to clipped destination
            px1 = x1 - rx
            py1 = y1 - ry
            px2 = px1 + (x2 - x1)
            py2 = py1 + (y2 - y1)

            sub_warped = warped_patch[py1:py2, px1:px2]
            sub_mask = m_view[py1:py2, px1:px2]
            sub_dst = dst[y1:y2, x1:x2]

            cv2.copyTo(sub_warped, sub_mask, sub_dst)

        return dst

    def warp(
        self,
        src_img: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
        dst_shape: Tuple[int, ...],
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Warps source face to target geometry using source and target landmarks.

        Args:
            src_img: Source image (BGR uint8).
            src_landmarks: Source landmarks (130, 2) or (>=468, 2).
            dst_landmarks: Target landmarks (130, 2) or (>=468, 2).
            dst_shape: Target canvas shape (height, width) or (height, width, 3).
            out: Optional pre-allocated output array.

        Returns:
            np.ndarray: Warped image of shape (height, width, 3).
        """
        # If this source is not cached or different image, cache it
        if self._cached_src_id != id(src_img) or self._cached_source_patches is None:
            self.set_source(src_img, src_landmarks)

        return self.warp_to_target(dst_landmarks, dst_shape, out=out)
