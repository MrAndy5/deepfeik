"""
deepfeik Android — Real-Time Face Swap
Kivy-based mobile app using the same core engine as the desktop version.

Entry point for Buildozer (main.py in android/ directory).
Includes crash logging, graceful error popups, and permission handling.
"""

import datetime
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Optional

# Determine all safe writable log paths on Android
_LOG_PATHS = []
try:
    # 1. App internal private data directory (always writable without permissions on Android)
    if "ANDROID_ARGUMENT" in os.environ or "PYTHONPATH" in os.environ:
        _LOG_PATHS.append(os.path.join(os.path.expanduser("~"), "deepfeik_crash.log"))
        _LOG_PATHS.append("/data/data/com.mrandy5.deepfeik/files/deepfeik_crash.log")
    # 2. Public storage
    _LOG_PATHS.append("/sdcard/deepfeik_crash.log")
    _LOG_PATHS.append("deepfeik_crash.log")
except Exception:
    _LOG_PATHS.append("deepfeik_crash.log")

def _log_to_file(msg: str):
    entry = f"[{datetime.datetime.now().isoformat()}] {msg}\n"
    for p in _LOG_PATHS:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass

_log_to_file("=== deepfeik Android starting ===")

def _android_excepthook(exc_type, exc_value, exc_tb):
    err = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _log_to_file(f"UNCAUGHT EXCEPTION:\n{err}")
    sys.stderr.write(err)
    sys.stderr.flush()

sys.excepthook = _android_excepthook

try:
    import cv2
    import numpy as np
    _log_to_file(f"OpenCV ({cv2.__version__}) and NumPy ({np.__version__}) loaded successfully.")
except Exception as e:
    _log_to_file(f"Error loading cv2/numpy: {e}")

# ── Kivy imports ──────────────────────────────────────────────────────────────
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

# Android-specific imports (available only when compiled with Buildozer)
try:
    from android.permissions import request_permissions, Permission
    from android import activity
    ANDROID = True
    _log_to_file("Running in ANDROID native environment.")
except ImportError:
    ANDROID = False
    _log_to_file("Running in desktop/emulated environment (ANDROID=False).")

# ── Core face swap engine ─────────────────────────────────────────────────────
try:
    from deepfeik.core.engine import FaceSwapEngine
    _log_to_file("FaceSwapEngine imported successfully.")
except Exception as e:
    _log_to_file(f"Error importing FaceSwapEngine: {traceback.format_exc()}")
    FaceSwapEngine = None


# ─────────────────────────────────────────────────────────────────────────────
# Camera texture capture (platform-aware)
# ─────────────────────────────────────────────────────────────────────────────

class CameraReader(threading.Thread):
    """Background thread reading frames from OpenCV camera."""

    def __init__(self, engine: Optional[FaceSwapEngine], on_frame_cb, on_error_cb):
        super().__init__(daemon=True)
        self._engine = engine
        self._on_frame = on_frame_cb
        self._on_error = on_error_cb
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self):
        self._running = True
        cam_index = 1 if ANDROID else 0  # Android front camera is usually index 1
        _log_to_file(f"Attempting to open camera index {cam_index}...")

        try:
            self._cap = cv2.VideoCapture(cam_index)
            if not self._cap or not self._cap.isOpened():
                _log_to_file(f"Camera index {cam_index} failed, trying index 0...")
                self._cap = cv2.VideoCapture(0)
        except Exception as e:
            _log_to_file(f"Exception opening camera: {e}")

        if not self._cap or not self._cap.isOpened():
            err_msg = "Cannot open camera. Check permissions in device settings."
            _log_to_file(err_msg)
            if self._on_error:
                Clock.schedule_once(lambda dt: self._on_error(err_msg), 0)
            return

        _log_to_file("Camera opened successfully.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        consecutive_errors = 0
        while self._running:
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 300:
                        _log_to_file("Camera stopped returning frames.")
                        break
                    continue

                consecutive_errors = 0
                frame = cv2.flip(frame, 1)  # mirror

                if self._engine is not None:
                    try:
                        processed = self._engine.process_frame(frame)
                    except Exception as e:
                        processed = frame
                else:
                    processed = frame

                self._on_frame(processed)
            except Exception as e:
                _log_to_file(f"Error in CameraReader loop: {e}")
                break

        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
        _log_to_file("Camera released.")

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
        self._engine: Optional[FaceSwapEngine] = None
        self._camera_reader: Optional[CameraReader] = None
        self._frame_texture: Optional[Texture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._build_ui()

        # Safely instantiate core engine
        try:
            if FaceSwapEngine is not None:
                self._engine = FaceSwapEngine()
                _log_to_file("FaceSwapEngine initialized.")
            else:
                self._show_error_popup("Core Engine Missing", "FaceSwapEngine could not be imported.")
        except Exception as e:
            err = traceback.format_exc()
            _log_to_file(f"Error initializing FaceSwapEngine: {err}")
            self._show_error_popup("Engine Init Error", f"{e}\n\nLog written to:\n{_LOG_PATHS[0]}")

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
            size_hint_x=0.5,
        )
        self._fps_lbl = Label(
            text="FPS: —",
            color=(*TEXT_DIM[:3], 1),
            font_size="13sp",
            size_hint_x=0.3,
        )
        btn_logs = _make_button("📋 Logs", size_hint_x=0.2, height="36dp")
        btn_logs.bind(on_press=self._show_logs_popup)

        title_bar.add_widget(lbl)
        title_bar.add_widget(self._fps_lbl)
        title_bar.add_widget(btn_logs)
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

    def _show_error_popup(self, title: str, message: str):
        content = BoxLayout(orientation="vertical", spacing=10, padding=12)
        scroll = ScrollView(size_hint=(1, 0.8))
        lbl = Label(
            text=message,
            color=(1, 0.4, 0.4, 1),
            font_size="12sp",
            size_hint_y=None,
        )
        lbl.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1]))
        scroll.add_widget(lbl)
        content.add_widget(scroll)

        btn_close = _make_button("Dismiss", height="44dp")
        popup = Popup(title=title, content=content, size_hint=(0.9, 0.7))
        btn_close.bind(on_press=popup.dismiss)
        content.add_widget(btn_close)
        popup.open()

    def _show_logs_popup(self, *args):
        log_content = ""
        found_path = ""
        for p in _LOG_PATHS:
            try:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        log_content = f.read()[-3000:]
                    found_path = p
                    break
            except Exception:
                pass
        if not log_content:
            log_content = "No logs recorded yet or storage pending permission."

        content = BoxLayout(orientation="vertical", spacing=8, padding=10)
        scroll = ScrollView(size_hint=(1, 0.85))
        lbl = Label(
            text=log_content,
            color=(0.9, 0.9, 0.9, 1),
            font_size="11sp",
            size_hint_y=None,
        )
        lbl.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1]))
        scroll.add_widget(lbl)
        content.add_widget(scroll)

        btn_close = _make_button("Close", height="44dp")
        popup = Popup(title=f"deepfeik Logs ({found_path or 'internal'})", content=content, size_hint=(0.95, 0.85))
        btn_close.bind(on_press=popup.dismiss)
        content.add_widget(btn_close)
        popup.open()

    def _request_permissions(self):
        if ANDROID:
            _log_to_file("Requesting Android runtime permissions...")
            perms = [
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ]
            if hasattr(Permission, "READ_MEDIA_IMAGES"):
                perms.append(Permission.READ_MEDIA_IMAGES)
            try:
                request_permissions(perms, self._on_permissions_granted)
            except Exception as e:
                _log_to_file(f"Error requesting permissions: {e}")
                self._start_camera()
        else:
            self._start_camera()

    def _on_permissions_granted(self, permissions, grants):
        _log_to_file(f"Permissions granted callback: {grants}")
        # If camera is granted, proceed
        if any(grants):
            self._start_camera()
        else:
            self._set_status("Camera permission denied.")

    def _start_camera(self):
        self._camera_reader = CameraReader(
            self._engine,
            self._on_new_frame,
            on_error_cb=lambda err: self._set_status(f"⚠️ {err}"),
        )
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

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.flip(rgb, 0)  # Kivy texture is flipped vertically
            h, w, _ = rgb.shape

            if self._frame_texture is None or self._frame_texture.width != w or self._frame_texture.height != h:
                self._frame_texture = Texture.create(size=(w, h), colorfmt="rgb")

            self._frame_texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            self._cam_image.texture = self._frame_texture
        except Exception as e:
            _log_to_file(f"Error updating display texture: {e}")

    def _on_load_face(self, *args):
        """Open file chooser to pick a face photo."""
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        initial_path = "/sdcard" if os.path.exists("/sdcard") else str(Path.home())
        fc = FileChooserListView(
            path=initial_path,
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
        if self._engine is None:
            self._set_status("Engine not loaded.")
            return
        ok, msg = self._engine.set_source_face(bgr)
        if not ok:
            self._set_status(f"No face detected: {msg}")
        else:
            self._btn_swap.text = "● Swap ON"
            self._btn_swap.background_color = (0.47, 0.15, 0.15, 1)
            self._set_status(f"✅ Face loaded — {os.path.basename(path)}")

    def _on_toggle_swap(self, *args):
        if self._engine and self._engine.is_source_loaded():
            self._engine.clear_source_face()
            self._btn_swap.text = "● Swap OFF"
            self._btn_swap.background_color = BTN_BG
            self._set_status("Swap disabled.")
        else:
            self._set_status("Load a face photo first.")

    def _on_clear(self, *args):
        if self._engine:
            self._engine.clear_source_face()
        self._btn_swap.text = "● Swap OFF"
        self._btn_swap.background_color = BTN_BG
        self._set_status("Face cleared.")

    def _set_status(self, msg: str):
        self._status.text = msg

    def on_stop(self):
        if self._camera_reader:
            self._camera_reader.stop()
        if self._engine:
            self._engine.close()
        _log_to_file("=== deepfeik Android stopped cleanly ===")


class DeepfeikApp(App):
    def build(self):
        return DeepfeikWidget()

    def on_stop(self):
        if hasattr(self.root, "on_stop"):
            self.root.on_stop()


if __name__ == "__main__":
    try:
        DeepfeikApp().run()
    except Exception as exc:
        _android_excepthook(*sys.exc_info())
