"""
deepfeik.gui.main_window

Modern, fully-resizable PyQt5 main window for the deepfeik real-time face swap app.

Design language: dark glassmorphism with a purple/blue accent palette.
Layout adapts fluidly from ~700×500 to fullscreen using stretch weights
and a scalable VideoWidget that fills available space.

Keyboard shortcuts:
  Ctrl+V  — paste image from clipboard
  Ctrl+O  — open file picker
  Ctrl+R  — toggle recording
  Escape  — clear source face
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QKeySequence, QFont, QColor,
    QPainter, QLinearGradient, QBrush, QPalette,
)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
    QStatusBar, QFrame, QSizePolicy, QApplication,
    QShortcut, QGraphicsDropShadowEffect,
)

from deepfeik.core.engine import FaceSwapEngine
from deepfeik.gui.camera_thread import CameraThread
from deepfeik.gui.clipboard import ClipboardService

logger = logging.getLogger(__name__)

_DEFAULT_SAVE_DIR = str(Path.home() / "Videos")

# ── Palette ────────────────────────────────────────────────────────────────────
_BG_DEEP   = "#0d0d14"   # near-black background
_BG_CARD   = "#14141f"   # slightly lighter card surface
_BG_HOVER  = "#1e1e30"   # hover state
_ACCENT    = "#7c5cbf"   # purple accent
_ACCENT2   = "#4a90d9"   # blue accent (gradient second stop)
_ACCENT_LT = "#9e7de0"   # lighter purple for text/glow
_TEXT      = "#e8e8f0"   # primary text
_TEXT_DIM  = "#7070a0"   # secondary / caption text
_SUCCESS   = "#4ec969"   # FPS green
_WARN      = "#e0a030"   # FPS orange
_ERROR     = "#e04040"   # FPS red / recording red

# ── Stylesheet ─────────────────────────────────────────────────────────────────
_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {_BG_DEEP};
    color: {_TEXT};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 12px;
}}

/* ── Card panel ────────────────────────────────── */
#sidePanel {{
    background-color: {_BG_CARD};
    border-left: 1px solid #1e1e32;
    border-radius: 0;
}}

/* ── Section labels ────────────────────────────── */
#sectionLabel {{
    color: {_ACCENT_LT};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 2px 0;
}}

/* ── FPS counter ───────────────────────────────── */
#fpsLabel {{
    font-size: 22px;
    font-weight: 700;
    color: {_SUCCESS};
}}

/* ── Thumbnail ─────────────────────────────────── */
#thumbLabel {{
    background-color: #0a0a12;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    color: {_TEXT_DIM};
    font-size: 11px;
}}

/* ── Primary action buttons ────────────────────── */
QPushButton {{
    background-color: #1a1a2e;
    color: {_TEXT};
    border: 1px solid #2a2a44;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 12px;
    font-weight: 500;
    text-align: left;
}}
QPushButton:hover {{
    background-color: {_BG_HOVER};
    border-color: {_ACCENT};
    color: {_ACCENT_LT};
}}
QPushButton:pressed {{
    background-color: #101020;
}}
QPushButton:disabled {{
    color: #3a3a5a;
    border-color: #1a1a2e;
}}

/* ── Accent load button ─────────────────────────── */
#loadBtn {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {_ACCENT}, stop:1 {_ACCENT2}
    );
    border: none;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 12px;
}}
#loadBtn:hover {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {_ACCENT_LT}, stop:1 #5aa0e8
    );
}}

/* ── Record button ──────────────────────────────── */
#recBtn {{
    border-color: #e04040;
    color: {_ERROR};
}}
#recBtn:hover {{
    background-color: #2a0000;
    border-color: {_ERROR};
}}
#recBtnActive {{
    background-color: {_ERROR};
    border: none;
    color: #ffffff;
    font-weight: 700;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 12px;
    text-align: left;
}}

/* ── Divider ────────────────────────────────────── */
#divider {{
    background-color: #1e1e32;
    max-height: 1px;
    border: none;
}}

/* ── Status bar ─────────────────────────────────── */
QStatusBar {{
    background-color: #0a0a12;
    color: {_TEXT_DIM};
    font-size: 11px;
    border-top: 1px solid #1a1a2e;
}}
"""


def _bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q)


class _VideoWidget(QLabel):
    """Scalable camera feed widget. Keeps aspect ratio and fills available space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(f"background-color: #080810; border: none;")
        self._pixmap: Optional[QPixmap] = None
        self._placeholder_text = "📷  Camera starting…"

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder_text = text
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(_BG_DEEP))

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor(_TEXT_DIM))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(), Qt.AlignCenter, self._placeholder_text
            )


def _make_section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionLabel")
    return lbl


def _make_divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    f.setFixedHeight(1)
    f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return f


class MainWindow(QMainWindow):
    """Main application window — modern dark glassmorphism design, fully resizable."""

    def __init__(self, device_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("deepfeik  •  Real-Time Face Swap")
        self.setMinimumSize(700, 480)
        self.resize(1100, 680)

        self._engine = FaceSwapEngine()
        self._camera_thread = CameraThread(
            engine=self._engine,
            device_index=device_index,
            width=1280,
            height=720,
        )
        self._camera_thread.frame_ready.connect(self._on_frame_ready)
        self._camera_thread.fps_updated.connect(self._on_fps_updated)
        self._camera_thread.camera_error.connect(self._on_camera_error)

        self._is_recording = False

        self._build_ui()
        self._install_shortcuts()

        self._camera_thread.start()
        self._set_status("Camera starting…  point your face at the webcam.")

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(_STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ─── Left: video feed ─────────────────────────────────────────────────
        self._video = _VideoWidget()
        layout.addWidget(self._video, stretch=10)

        # ─── Right: side panel ────────────────────────────────────────────────
        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(240)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 20, 16, 16)
        panel_layout.setSpacing(12)

        # App title
        title = QLabel("deepfeik")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 800; color: {_ACCENT_LT}; letter-spacing: 1px;"
        )
        panel_layout.addWidget(title)

        subtitle = QLabel("Real-Time Face Swap")
        subtitle.setStyleSheet(f"font-size: 10px; color: {_TEXT_DIM}; margin-top: -8px;")
        panel_layout.addWidget(subtitle)

        panel_layout.addWidget(_make_divider())

        # ─── FPS ─────────────────────────────────────────────────────────────
        fps_row = QHBoxLayout()
        fps_icon = QLabel("⚡")
        fps_icon.setStyleSheet("font-size: 14px;")
        self._fps_label = QLabel("—  FPS")
        self._fps_label.setObjectName("fpsLabel")
        fps_row.addWidget(fps_icon)
        fps_row.addWidget(self._fps_label)
        fps_row.addStretch()
        panel_layout.addLayout(fps_row)

        panel_layout.addWidget(_make_divider())

        # ─── Source face ──────────────────────────────────────────────────────
        panel_layout.addWidget(_make_section_label("SOURCE FACE"))

        self._thumb = QLabel("No face loaded")
        self._thumb.setObjectName("thumbLabel")
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setFixedHeight(150)
        self._thumb.setWordWrap(True)
        panel_layout.addWidget(self._thumb)

        self._btn_load = QPushButton("📂  Open Image…")
        self._btn_load.setObjectName("loadBtn")
        self._btn_load.setToolTip("Browse for a face photo (Ctrl+O)")
        self._btn_load.clicked.connect(self._on_load)
        panel_layout.addWidget(self._btn_load)

        self._btn_paste = QPushButton("📋  Paste from Clipboard")
        self._btn_paste.setToolTip("Paste a copied image (Ctrl+V)")
        self._btn_paste.clicked.connect(self._on_paste)
        panel_layout.addWidget(self._btn_paste)

        self._btn_clear = QPushButton("✖  Clear Face")
        self._btn_clear.setToolTip("Remove source face (Esc)")
        self._btn_clear.setEnabled(False)
        self._btn_clear.clicked.connect(self._on_clear)
        panel_layout.addWidget(self._btn_clear)

        panel_layout.addWidget(_make_divider())

        # ─── Recording ────────────────────────────────────────────────────────
        panel_layout.addWidget(_make_section_label("RECORDING"))

        self._btn_rec = QPushButton("⏺  Start Recording")
        self._btn_rec.setObjectName("recBtn")
        self._btn_rec.setToolTip("Record the live swapped video (Ctrl+R)")
        self._btn_rec.clicked.connect(self._on_toggle_rec)
        panel_layout.addWidget(self._btn_rec)

        self._rec_badge = QLabel("")
        self._rec_badge.setAlignment(Qt.AlignCenter)
        self._rec_badge.setStyleSheet(f"font-size: 10px; color: {_TEXT_DIM};")
        panel_layout.addWidget(self._rec_badge)

        panel_layout.addStretch()

        # ─── Tip ──────────────────────────────────────────────────────────────
        tip = QLabel("Tip: Load a clear front-facing\nphoto for best results.")
        tip.setAlignment(Qt.AlignCenter)
        tip.setWordWrap(True)
        tip.setStyleSheet(f"font-size: 10px; color: {_TEXT_DIM}; line-height: 150%;")
        panel_layout.addWidget(tip)

        layout.addWidget(panel)

        # ─── Status bar ───────────────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+V"), self).activated.connect(self._on_paste)
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self._on_load)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self._on_toggle_rec)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._on_clear)

    # ── Slots ──────────────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        self._video.set_frame_pixmap(_bgr_to_qpixmap(frame))

    @pyqtSlot(float)
    def _on_fps_updated(self, fps: float) -> None:
        color = _SUCCESS if fps >= 15 else (_WARN if fps >= 8 else _ERROR)
        self._fps_label.setText(f"{fps:.0f}  FPS")
        self._fps_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {color};"
        )

    @pyqtSlot(str)
    def _on_camera_error(self, msg: str) -> None:
        self._video.set_placeholder(f"⚠️  {msg}")
        self._set_status(f"Camera error: {msg}", error=True)
        QMessageBox.critical(self, "Camera Error", msg)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Face Image",
            str(Path.home() / "Pictures"),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
        )
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Load Error", f"Could not read:\n{path}")
            return
        self._apply_source(bgr, os.path.basename(path))

    def _on_paste(self) -> None:
        bgr, err = ClipboardService.get_image_from_clipboard()
        if bgr is None:
            QMessageBox.warning(self, "Paste Error", err or "No image on clipboard.")
            return
        self._apply_source(bgr, "clipboard")

    def _on_clear(self) -> None:
        self._engine.clear_source_face()
        self._thumb.setText("No face loaded")
        self._thumb.setPixmap(QPixmap())
        self._btn_clear.setEnabled(False)
        self._set_status("Source face cleared — passthrough active.")

    def _on_toggle_rec(self) -> None:
        if not self._is_recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(_DEFAULT_SAVE_DIR, exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Recording As",
                os.path.join(_DEFAULT_SAVE_DIR, f"deepfeik_{ts}.mp4"),
                "MP4 Video (*.mp4);;AVI Video (*.avi)",
            )
            if not path:
                return
            ok = self._camera_thread.start_recording(path)
            if not ok:
                QMessageBox.critical(self, "Recording Error", f"Could not record to:\n{path}")
                return
            self._is_recording = True
            self._btn_rec.setText("⏹  Stop Recording")
            self._btn_rec.setObjectName("recBtnActive")
            self._btn_rec.setStyle(self._btn_rec.style())  # force style refresh
            self._rec_badge.setText("● REC")
            self._rec_badge.setStyleSheet(
                f"font-size: 11px; font-weight: 700; color: {_ERROR};"
            )
            self._set_status(f"Recording → {os.path.basename(path)}")
        else:
            saved = self._camera_thread.stop_recording()
            self._is_recording = False
            self._btn_rec.setText("⏺  Start Recording")
            self._btn_rec.setObjectName("recBtn")
            self._btn_rec.setStyle(self._btn_rec.style())
            self._rec_badge.setText("")
            self._rec_badge.setStyleSheet(f"font-size: 10px; color: {_TEXT_DIM};")
            if saved:
                self._set_status(f"Saved: {saved}")
                QMessageBox.information(self, "Recording Saved", f"Video saved:\n{saved}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _apply_source(self, bgr: np.ndarray, label: str) -> None:
        ok, msg = self._engine.set_source_face(bgr)
        if not ok:
            QMessageBox.warning(
                self, "No Face Detected",
                f"No face found in the loaded image.\n\n{msg}\n\n"
                "Try a clear, front-facing photo."
            )
            return
        th = self._thumb.height() - 4
        tw = self._thumb.width() - 4
        thumb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
        self._thumb.setPixmap(_bgr_to_qpixmap(thumb))
        self._btn_clear.setEnabled(True)
        self._set_status(f"✅  Face loaded from {label}  —  swap active!")

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status.showMessage(msg)
        col = _ERROR if error else _TEXT_DIM
        self._status.setStyleSheet(f"color: {col};")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._is_recording:
            self._camera_thread.stop_recording()
        self._camera_thread.stop()
        self._camera_thread.wait(3000)
        self._engine.close()
        event.accept()
