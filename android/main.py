"""
deepfeik Android — Real-Time Face Swap
Kivy-based mobile app using the same core engine as the desktop version.

Entry point for Buildozer (main.py in android/ directory).
"""

import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Kivy imports ──────────────────────────────────────────────────────────────
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

# Android-specific imports (available only when compiled with Buildozer)
try:
    from android.permissions import request_permissions, Permission
    from android import activity
    ANDROID = True
except ImportError:
    ANDROID = False

# ── Core face swap engine ─────────────────────────────────────────────────────
# Same engine as desktop — zero GUI dependencies
from deepfeik.core.engine import FaceSwapEngine


# ─────────────────────────────────────────────────────────────────────────────
# Camera texture capture (platform-aware)
# ─────────────────────────────────────────────────────────────────────────────

class CameraReader(threading.Thread):
    """Background thread reading frames from OpenCV camera."""

    def __init__(self, engine: FaceSwapEngine, on_frame_cb):
        super().__init__(daemon=True)
        self._engine = engine
        self._on_frame = on_frame_cb
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self):
        self._running = True
        cam_index = 1 if ANDROID else 0  # Android front camera is usually index 1
        self._cap = cv2.VideoCapture(cam_index)
        if not self._cap.isOpened():
            # Fallback to index 0
            self._cap = cv2.VideoCapture(0)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)  # mirror

            try:
                processed = self._engine.process_frame(frame)
            except Exception:
                processed = frame

            self._on_frame(processed)

        if self._cap:
            self._cap.release()

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

DARK = (0.05, 0.05, 0.08, 1)
ACCENT = (0.47, 0.36, 0.75, 1)
BTN_BG = (0.1, 0.1, 0.16, 1)
TEXT_DIM = (0.44, 0.44, 0.63, 1)


def _make_button(text: str, **kwargs) -> Button:
    btn = Button(
        text=text,
        background_color=BTN_BG,
        background_normal="",
        color=(0.9, 0.9, 1, 1),
        font_size="14sp",
        size_hint_y=None,
        height="48dp",
        **kwargs,
    )
    return btn


class DeepfeikWidget(FloatLayout):
    """Root UI widget."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = FaceSwapEngine()
        self._camera_reader: Optional[CameraReader] = None
        self._frame_texture: Optional[Texture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._build_ui()
        self._request_permissions()

    def _build_ui(self):
        Window.clearcolor = DARK

        # Main vertical layout
        layout = BoxLayout(orientation="vertical", padding=12, spacing=8)

        # Title bar
        title_bar = BoxLayout(size_hint_y=None, height="44dp", spacing=8)
        lbl = Label(
            text="deepfeik",
            color=(*ACCENT[:3], 1),
            font_size="20sp",
            bold=True,
            size_hint_x=0.6,
        )
        self._fps_lbl = Label(
            text="FPS: —",
            color=(*TEXT_DIM[:3], 1),
            font_size="13sp",
            size_hint_x=0.4,
        )
        title_bar.add_widget(lbl)
        title_bar.add_widget(self._fps_lbl)
        layout.add_widget(title_bar)

        # Camera preview
        self._cam_image = KivyImage(
            allow_stretch=True,
            keep_ratio=True,
        )
        layout.add_widget(self._cam_image)

        # Controls row
        ctrl_row = BoxLayout(size_hint_y=None, height="56dp", spacing=10)

        btn_load = _make_button("📂 Load Face")
        btn_load.bind(on_press=self._on_load_face)
        ctrl_row.add_widget(btn_load)

        self._btn_swap = _make_button("● Swap OFF", background_color=(0.1, 0.1, 0.16, 1))
        self._btn_swap.bind(on_press=self._on_toggle_swap)
        ctrl_row.add_widget(self._btn_swap)

        btn_clear = _make_button("✖ Clear")
        btn_clear.bind(on_press=self._on_clear)
        ctrl_row.add_widget(btn_clear)

        layout.add_widget(ctrl_row)

        # Status bar
        self._status = Label(
            text="Tap '📂 Load Face' to load a source photo",
            color=(*TEXT_DIM[:3], 1),
            font_size="11sp",
            size_hint_y=None,
            height="28dp",
        )
        layout.add_widget(self._status)

        self.add_widget(layout)

    def _request_permissions(self):
        if ANDROID:
            request_permissions([
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ], self._on_permissions_granted)
        else:
            self._start_camera()

    def _on_permissions_granted(self, permissions, grants):
        if all(grants):
            self._start_camera()
        else:
            self._set_status("Camera permission denied.")

    def _start_camera(self):
        self._camera_reader = CameraReader(self._engine, self._on_new_frame)
        self._camera_reader.start()
        Clock.schedule_interval(self._update_display, 1.0 / 30.0)
        self._set_status("Camera active. Load a face photo to begin swap.")

    def _on_new_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame

    def _update_display(self, dt):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 0)  # Kivy texture is flipped vertically
        h, w, _ = rgb.shape

        if self._frame_texture is None or self._frame_texture.width != w or self._frame_texture.height != h:
            self._frame_texture = Texture.create(size=(w, h), colorfmt="rgb")

        self._frame_texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self._cam_image.texture = self._frame_texture

    def _on_load_face(self, *args):
        """Open file chooser to pick a face photo."""
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        fc = FileChooserListView(
            path=str(Path.home()),
            filters=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"],
        )
        btn_row = BoxLayout(size_hint_y=None, height="48dp", spacing=8)
        popup = Popup(title="Select Face Photo", content=content, size_hint=(0.95, 0.85))

        def _load(instance):
            if fc.selection:
                path = fc.selection[0]
                self._load_image_path(path)
            popup.dismiss()

        btn_ok = _make_button("Load")
        btn_ok.bind(on_press=_load)
        btn_cancel = _make_button("Cancel")
        btn_cancel.bind(on_press=lambda x: popup.dismiss())

        btn_row.add_widget(btn_ok)
        btn_row.add_widget(btn_cancel)
        content.add_widget(fc)
        content.add_widget(btn_row)
        popup.open()

    def _load_image_path(self, path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self._set_status(f"Could not read: {os.path.basename(path)}")
            return
        ok, msg = self._engine.set_source_face(bgr)
        if not ok:
            self._set_status(f"No face detected: {msg}")
        else:
            self._btn_swap.text = "● Swap ON"
            self._btn_swap.background_color = (0.47, 0.15, 0.15, 1)
            self._set_status(f"✅ Face loaded — {os.path.basename(path)}")

    def _on_toggle_swap(self, *args):
        if self._engine.is_source_loaded():
            self._engine.clear_source_face()
            self._btn_swap.text = "● Swap OFF"
            self._btn_swap.background_color = BTN_BG
            self._set_status("Swap disabled.")
        else:
            self._set_status("Load a face photo first.")

    def _on_clear(self, *args):
        self._engine.clear_source_face()
        self._btn_swap.text = "● Swap OFF"
        self._btn_swap.background_color = BTN_BG
        self._set_status("Face cleared.")

    def _set_status(self, msg: str):
        self._status.text = msg

    def on_stop(self):
        if self._camera_reader:
            self._camera_reader.stop()
        self._engine.close()


class DeepfeikApp(App):
    def build(self):
        return DeepfeikWidget()

    def on_stop(self):
        if hasattr(self.root, "on_stop"):
            self.root.on_stop()


if __name__ == "__main__":
    DeepfeikApp().run()
