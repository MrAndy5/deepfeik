"""
deepfeik.gui.clipboard

System clipboard image extraction service supporting direct bitmap pixels
and copied image files from Windows Explorer (Qt primary with Pillow fallback).
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np

# Valid image extensions to check when clipboard contains copied file paths
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ClipboardService:
    """Service for extracting and validating images from the system clipboard."""

    @staticmethod
    def get_image_from_clipboard() -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extracts image from system clipboard (bitmap or copied file).

        Attempts extraction via PyQt5 QClipboard if an application instance exists.
        Falls back to Pillow ImageGrab only when no Qt application instance is active.

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: (bgr_image, error_message).
            If successful, bgr_image is a BGR uint8 NumPy array and error_message is None.
            If failed, bgr_image is None and error_message describes the reason.
        """
        # 1. Attempt extraction via Qt clipboard if QApplication is active
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                return ClipboardService._get_from_qt_clipboard(app)
        except Exception:
            pass

        # 2. Fallback to Pillow ImageGrab if no Qt application is running
        return ClipboardService._get_from_pillow()

    @staticmethod
    def _get_from_qt_clipboard(app) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extracts image using PyQt5 QClipboard in a non-blocking manner."""
        from PyQt5.QtGui import QImage

        clipboard = app.clipboard()
        mime = clipboard.mimeData()

        if mime is None:
            return None, "Clipboard is empty."

        formats = mime.formats()
        if not formats:
            return None, "Clipboard is empty."

        # Case A: Direct QImage / Pixmap bitmap
        if mime.hasImage():
            qimg = clipboard.image()
            if not qimg.isNull():
                qimg_rgba = qimg.convertToFormat(QImage.Format_RGBA8888)
                w = qimg_rgba.width()
                h = qimg_rgba.height()
                if w > 0 and h > 0:
                    ptr = qimg_rgba.bits()
                    ptr.setsize(h * w * 4)
                    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    return bgr, None

        # Case B: Copied file URLs from Windows Explorer (e.g. Ctrl+C on a jpg)
        if mime.hasUrls():
            for url in mime.urls():
                file_path = url.toLocalFile()
                if file_path and os.path.isfile(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        if bgr is not None and bgr.size > 0:
                            return bgr, None
            return None, "No valid image files found on clipboard."

        # Case C: Plain text on clipboard
        if mime.hasText():
            text = mime.text().strip()
            # Check if text is a file path to an image
            if text and os.path.isfile(text):
                ext = os.path.splitext(text)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    bgr = cv2.imread(text, cv2.IMREAD_COLOR)
                    if bgr is not None and bgr.size > 0:
                        return bgr, None
            return None, "Clipboard contains text, not an image. Please copy an image or image file."

        return None, "Clipboard is empty or does not contain a valid image."

    @staticmethod
    def _get_from_pillow() -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extracts image using Pillow ImageGrab (used in non-Qt environments)."""
        try:
            from PIL import ImageGrab, Image

            data = ImageGrab.grabclipboard()
            if data is None:
                return None, "Clipboard is empty."

            # Case A: PIL Image object (direct bitmap)
            if isinstance(data, Image.Image):
                rgb_img = data.convert("RGB")
                arr = np.array(rgb_img, dtype=np.uint8)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return bgr, None

            # Case B: List of file paths (Explorer copy)
            if isinstance(data, list):
                for p in data:
                    if isinstance(p, str) and os.path.isfile(p):
                        ext = os.path.splitext(p)[1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                            if bgr is not None and bgr.size > 0:
                                return bgr, None
                return None, "No valid image files found in clipboard file list."

            return None, "Clipboard does not contain image data."

        except Exception as e:
            return None, f"Clipboard error: {e}"
