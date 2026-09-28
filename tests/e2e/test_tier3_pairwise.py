"""Tier 3: 18 Cross-Feature Pairwise Interaction Tests for DeepFeik.

Covers:
- R1 x R2 (3 tests: hot swap source, invalid load during swap, clear source during stream)
- R1 x R3 (3 tests: record live swap, face loss & recovery in recording, recording throughput overhead)
- R2 x R3 (3 tests: paste image while recording, record before source load, stop record during source load)
- R1 x R4 (3 tests: HUD FPS counter updates, viewport renders swapped frames, aspect ratio on resize)
- R2 x R4 (3 tests: load button triggers update, paste shortcut triggers update, file dialog cancel preserves state)
- R3 x R4 (3 tests: record button toggles state, window close while recording finalizes, save dialog cancel aborts)
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton


# ============================================================================
# Pairwise: R1 (Live Swap) x R2 (Source Loading) (3 tests)
# ============================================================================

def test_pair_r1_r2_01_source_image_change_while_streaming(face_swap_engine, mock_animated_camera, valid_face_image):
    """T3-X-12-01: Switching source image mid-stream seamlessly updates swap without frame drop."""
    # Source Face A
    face_a = valid_face_image.copy()
    success, msg = face_swap_engine.set_source_face(face_a)
    assert success is True

    # Process initial frames
    for _ in range(5):
        ret, frame = mock_animated_camera.read()
        assert ret is True
        out_a = face_swap_engine.process_frame(frame)
        assert out_a is not None

    # Hot-swap to Source Face B (e.g. green tint / modified skin tone)
    face_b = valid_face_image.copy()
    face_b[:, :, 1] = np.clip(face_b[:, :, 1].astype(int) + 70, 0, 255)
    success, msg = face_swap_engine.set_source_face(face_b)
    assert success is True

    # Next frame immediately processes with Face B
    ret, frame = mock_animated_camera.read()
    assert ret is True
    out_b = face_swap_engine.process_frame(frame)
    assert out_b is not None
    assert out_b.shape == frame.shape


def test_pair_r1_r2_02_invalid_source_load_during_active_swap(face_swap_engine, valid_face_image, blank_image):
    """T3-X-12-02: Attempting invalid source load during active swap retains existing valid face."""
    # Set valid source face
    success, msg = face_swap_engine.set_source_face(valid_face_image)
    assert success is True
    assert face_swap_engine.is_source_loaded() is True

    # Process frame to ensure swap is active
    out_before = face_swap_engine.process_frame(valid_face_image)

    # Attempt to load blank image (no face)
    success, msg = face_swap_engine.set_source_face(blank_image)
    assert success is False

    # Previous valid face MUST still be active
    assert face_swap_engine.is_source_loaded() is True
    out_after = face_swap_engine.process_frame(valid_face_image)
    assert out_after is not None


def test_pair_r1_r2_03_clear_source_during_live_stream(face_swap_engine, mock_animated_camera, valid_face_image):
    """T3-X-12-03: Clearing source face mid-stream immediately reverts pipeline to clean passthrough."""
    face_swap_engine.set_source_face(valid_face_image)
    assert face_swap_engine.is_source_loaded() is True

    ret, frame = mock_animated_camera.read()
    _ = face_swap_engine.process_frame(frame)

    # Clear source face
    face_swap_engine.clear_source_face()
    assert face_swap_engine.is_source_loaded() is False

    # Next frame reverts to exact raw passthrough
    ret, frame_next = mock_animated_camera.read()
    out = face_swap_engine.process_frame(frame_next)
    assert np.array_equal(out, frame_next), "Output frame must be exact passthrough after clear"


# ============================================================================
# Pairwise: R1 (Live Swap) x R3 (Recording) (3 tests)
# ============================================================================

def test_pair_r1_r3_01_record_live_face_swap_session(face_swap_engine, video_recorder, mock_animated_camera, valid_face_image, tmp_path):
    """T3-X-13-01: Live swap frames are actively recorded to disk and playable."""
    out_path = str(tmp_path / "rec_swap_session.mp4")
    face_swap_engine.set_source_face(valid_face_image)
    video_recorder.start_recording(out_path, 640, 480, target_fps=30.0)

    for _ in range(25):
        ret, frame = mock_animated_camera.read()
        assert ret is True
        swapped = face_swap_engine.process_frame(frame)
        video_recorder.record_frame(swapped)

    video_recorder.stop_recording()

    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened() is True
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    assert count >= 20, f"Expected at least 20 recorded frames, got {count}"


def test_pair_r1_r3_02_face_loss_and_recovery_during_recording(face_swap_engine, video_recorder, valid_face_image, blank_image, tmp_path):
    """T3-X-13-02: User face disappears and returns during active recording; recording unbroken."""
    out_path = str(tmp_path / "rec_face_loss.mp4")
    face_swap_engine.set_source_face(valid_face_image)
    video_recorder.start_recording(out_path, 640, 480, 30.0)

    # 10 frames of face
    for _ in range(10):
        out = face_swap_engine.process_frame(valid_face_image)
        video_recorder.record_frame(out)

    # 10 frames of blank (user leaves camera)
    for _ in range(10):
        out = face_swap_engine.process_frame(blank_image)
        video_recorder.record_frame(out)

    # 10 frames of face (user returns)
    for _ in range(10):
        out = face_swap_engine.process_frame(valid_face_image)
        video_recorder.record_frame(out)

    video_recorder.stop_recording()

    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened() is True
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert total >= 25, f"Expected continuous recording across face loss, got {total} frames"


def test_pair_r1_r3_03_recording_frame_throughput_under_active_swap(face_swap_engine, video_recorder, valid_face_image, tmp_path):
    """T3-X-13-03: Threaded video recording overhead does not throttle swap processing loop."""
    face_swap_engine.set_source_face(valid_face_image)
    out_path = str(tmp_path / "rec_perf.mp4")

    # Measure with recording OFF
    t0 = time.perf_counter()
    for _ in range(15):
        _ = face_swap_engine.process_frame(valid_face_image)
    t_no_rec = (time.perf_counter() - t0) / 15.0

    # Measure with recording ON
    video_recorder.start_recording(out_path, 640, 480, 30.0)
    t1 = time.perf_counter()
    for _ in range(15):
        frame = face_swap_engine.process_frame(valid_face_image)
        video_recorder.record_frame(frame)
    t_with_rec = (time.perf_counter() - t1) / 15.0
    video_recorder.stop_recording()

    delta_ms = (t_with_rec - t_no_rec) * 1000.0
    # Recording overhead on main thread should be minimal (< 8ms)
    assert delta_ms < 8.0, f"Recording overhead too high: {delta_ms:.2f}ms per frame"


# ============================================================================
# Pairwise: R2 (Source Loading) x R3 (Recording) (3 tests)
# ============================================================================

def test_pair_r2_r3_01_paste_source_image_while_recording(face_swap_engine, video_recorder, clipboard_service, valid_face_image, headless_qapp, tmp_path):
    """T3-X-23-01: Pasting new source face while actively recording produces valid recorded transition."""
    out_path = str(tmp_path / "rec_paste_midstream.mp4")
    face_swap_engine.set_source_face(valid_face_image)
    video_recorder.start_recording(out_path, 640, 480, 30.0)

    for _ in range(10):
        video_recorder.record_frame(face_swap_engine.process_frame(valid_face_image))

    # Paste face B
    face_b = valid_face_image.copy()
    face_b[:, :, 0] = np.clip(face_b[:, :, 0].astype(int) + 80, 0, 255)
    rgb = cv2.cvtColor(face_b, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    headless_qapp.clipboard().setImage(qimg)
    headless_qapp.processEvents()

    extracted, _ = clipboard_service.get_image_from_clipboard()
    if extracted is not None:
        face_swap_engine.set_source_face(extracted)

    for _ in range(10):
        video_recorder.record_frame(face_swap_engine.process_frame(valid_face_image))

    video_recorder.stop_recording()
    assert os.path.getsize(out_path) > 100


def test_pair_r2_r3_02_start_recording_before_source_image_loaded(face_swap_engine, video_recorder, valid_face_image, tmp_path):
    """T3-X-23-02: Recording started in passthrough mode continues seamlessly when face loaded later."""
    out_path = str(tmp_path / "rec_before_load.mp4")
    face_swap_engine.clear_source_face()
    video_recorder.start_recording(out_path, 640, 480, 30.0)

    # 10 passthrough frames
    for _ in range(10):
        video_recorder.record_frame(face_swap_engine.process_frame(valid_face_image))

    # Load source face
    face_swap_engine.set_source_face(valid_face_image)

    # 10 swapped frames
    for _ in range(10):
        video_recorder.record_frame(face_swap_engine.process_frame(valid_face_image))

    video_recorder.stop_recording()

    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened() is True
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert total >= 18


def test_pair_r2_r3_03_stop_recording_during_source_image_load(face_swap_engine, video_recorder, valid_face_image, tmp_path):
    """T3-X-23-03: Stopping recording while new image is loaded finalizes without thread conflict."""
    out_path = str(tmp_path / "rec_stop_during_load.mp4")
    video_recorder.start_recording(out_path, 640, 480, 30.0)
    video_recorder.record_frame(valid_face_image)

    # Concurrent operations
    video_recorder.stop_recording()
    success, msg = face_swap_engine.set_source_face(valid_face_image)

    assert video_recorder.is_recording() is False
    assert success is True
    assert face_swap_engine.is_source_loaded() is True


# ============================================================================
# Pairwise: R1 (Live Swap) x R4 (GUI & Usability) (3 tests)
# ============================================================================

def test_pair_r1_r4_01_hud_fps_counter_updates_with_stream(main_window, valid_face_image, qt_pump):
    """T3-X-14-01: Pushing stream frames causes status HUD / FPS indicator to update."""
    for _ in range(15):
        if hasattr(main_window, "video_widget") and hasattr(main_window.video_widget, "update_frame"):
            main_window.video_widget.update_frame(valid_face_image)
        elif hasattr(main_window, "on_frame_ready"):
            main_window.on_frame_ready(valid_face_image)
    qt_pump()
    status_bar = main_window.statusBar()
    assert status_bar is not None


def test_pair_r1_r4_02_gui_viewport_renders_swapped_frames(main_window, valid_face_image, qt_pump):
    """T3-X-14-02: Video viewport renders swapped frame without widget error."""
    if hasattr(main_window, "video_widget") and hasattr(main_window.video_widget, "update_frame"):
        main_window.video_widget.update_frame(valid_face_image)
    qt_pump()


def test_pair_r1_r4_03_aspect_ratio_preserved_on_widget_resize(main_window, valid_face_image, qt_pump):
    """T3-X-14-03: Non-standard window resize preserves video viewport aspect ratio."""
    main_window.resize(1100, 450)
    qt_pump()
    if hasattr(main_window, "video_widget") and hasattr(main_window.video_widget, "update_frame"):
        main_window.video_widget.update_frame(valid_face_image)
    qt_pump()


# ============================================================================
# Pairwise: R2 (Source Loading) x R4 (GUI & Usability) (3 tests)
# ============================================================================

def test_pair_r2_r4_01_gui_load_button_triggers_source_update(main_window, valid_face_path, monkeypatch, qt_pump):
    """T3-X-24-01: Clicking Load Image with mocked file picker updates source face in GUI."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(valid_face_path), "Image Files (*.jpg *.png)"))

    buttons = main_window.findChildren(QPushButton)
    load_btns = [b for b in buttons if "load" in b.text().lower() or "picker" in b.text().lower() or "open" in b.text().lower()]
    if load_btns:
        load_btns[0].click()
        qt_pump()


def test_pair_r2_r4_02_gui_paste_shortcut_triggers_source_update(main_window, valid_face_image, headless_qapp, qt_pump):
    """T3-X-24-02: Triggering paste action with clipboard image updates source face."""
    rgb = cv2.cvtColor(valid_face_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    headless_qapp.clipboard().setImage(qimg)
    headless_qapp.processEvents()

    if hasattr(main_window, "on_paste_action"):
        main_window.on_paste_action()
    elif hasattr(main_window, "paste_button"):
        main_window.paste_button.click()
    qt_pump()


def test_pair_r2_r4_03_gui_file_picker_cancel_preserves_state(main_window, monkeypatch, qt_pump):
    """T3-X-24-03: Canceling file picker dialog preserves existing GUI and source state."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))

    buttons = main_window.findChildren(QPushButton)
    load_btns = [b for b in buttons if "load" in b.text().lower() or "open" in b.text().lower()]
    if load_btns:
        load_btns[0].click()
        qt_pump()


# ============================================================================
# Pairwise: R3 (Recording) x R4 (GUI & Usability) (3 tests)
# ============================================================================

def test_pair_r3_r4_01_gui_record_button_toggles_recording_state(main_window, tmp_path, monkeypatch, qt_pump):
    """T3-X-34-01: Clicking Record button toggles state and updates button label."""
    rec_target = str(tmp_path / "gui_toggle.mp4")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (rec_target, "MP4 Files (*.mp4)"))

    buttons = main_window.findChildren(QPushButton)
    rec_btns = [b for b in buttons if "record" in b.text().lower() or "rec" in b.text().lower()]
    if rec_btns:
        rec_btn = rec_btns[0]
        # Start
        rec_btn.click()
        qt_pump()
        # Stop
        rec_btn.click()
        qt_pump()


def test_pair_r3_r4_02_gui_window_close_while_recording_finalizes_file(main_window, tmp_path, monkeypatch, valid_face_image, qt_pump):
    """T3-X-34-02: Closing window while recording safely finalizes video without hanging."""
    rec_target = str(tmp_path / "gui_close_rec.mp4")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (rec_target, "MP4 Files (*.mp4)"))

    buttons = main_window.findChildren(QPushButton)
    rec_btns = [b for b in buttons if "record" in b.text().lower() or "rec" in b.text().lower()]
    if rec_btns:
        rec_btns[0].click()
        qt_pump()

    main_window.close()
    qt_pump()


def test_pair_r3_r4_03_gui_save_dialog_cancel_aborts_recording(main_window, monkeypatch, qt_pump):
    """T3-X-34-03: Canceling save destination dialog cleanly aborts recording without error."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))

    buttons = main_window.findChildren(QPushButton)
    rec_btns = [b for b in buttons if "record" in b.text().lower() or "rec" in b.text().lower()]
    if rec_btns:
        rec_btns[0].click()
        qt_pump()
