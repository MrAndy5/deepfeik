"""
Unit tests for ClipboardService.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from deepfeik.gui.clipboard import ClipboardService
from tests.conftest import create_synthetic_face_image


def test_clipboard_empty(headless_qapp):
    """Verifies that an empty clipboard returns (None, error_msg)."""
    headless_qapp.clipboard().clear()
    headless_qapp.processEvents()

    img, err = ClipboardService.get_image_from_clipboard()
    assert img is None
    assert err is not None and "empty" in err.lower()


def test_clipboard_plain_text(headless_qapp):
    """Verifies that non-image plain text is rejected with an informative error message."""
    headless_qapp.clipboard().setText("Just some regular text string, not an image")
    headless_qapp.processEvents()

    img, err = ClipboardService.get_image_from_clipboard()
    assert img is None
    assert err is not None
    assert "text" in err.lower() or "image" in err.lower()


def test_clipboard_qt_image(headless_qapp):
    """Verifies extracting a direct bitmap image from Qt clipboard."""
    expected_bgr = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

    headless_qapp.clipboard().setImage(qimg)
    headless_qapp.processEvents()

    extracted, err = ClipboardService.get_image_from_clipboard()
    assert extracted is not None
    assert err is None
    assert extracted.shape == expected_bgr.shape
    assert np.array_equal(extracted, expected_bgr)


def test_clipboard_copied_file_url(headless_qapp, tmp_path: Path):
    """Verifies extracting an image when copied as a file URL (Explorer Ctrl+C)."""
    img_path = tmp_path / "test_copy.jpg"
    test_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(img_path), test_img)

    from PyQt5.QtCore import QMimeData
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(img_path))])
    headless_qapp.clipboard().setMimeData(mime)
    headless_qapp.processEvents()

    extracted, err = ClipboardService.get_image_from_clipboard()
    assert extracted is not None
    assert err is None
    assert extracted.shape == (100, 100, 3)


def test_clipboard_non_image_file_url(headless_qapp, tmp_path: Path):
    """Verifies copying a non-image file (e.g. .txt) is safely rejected."""
    txt_path = tmp_path / "document.txt"
    txt_path.write_text("Hello world")

    from PyQt5.QtCore import QMimeData
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(txt_path))])
    headless_qapp.clipboard().setMimeData(mime)
    headless_qapp.processEvents()

    extracted, err = ClipboardService.get_image_from_clipboard()
    assert extracted is None
    assert err is not None


def test_clipboard_pillow_fallback(tmp_path: Path):
    """Verifies Pillow-based fallback extractor logic."""
    from PIL import Image
    test_pil = Image.new("RGB", (60, 60), color=(100, 150, 200))

    # Mock ImageGrab.grabclipboard
    from unittest.mock import patch
    with patch("PIL.ImageGrab.grabclipboard", return_value=test_pil):
        extracted, err = ClipboardService._get_from_pillow()
        assert extracted is not None
        assert err is None
        assert extracted.shape == (60, 60, 3)
