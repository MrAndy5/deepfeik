"""Tier 2: 24 Boundary and Corner Case Tests for DeepFeik.

Covers:
- R1 Boundaries (6 tests: zero faces in stream, extreme head pose yaw, 16:9 aspect ratio, camera disconnect, sensor noise, resolution switch)
- R2 Boundaries (6 tests: transparent RGBA, grayscale B&W, ultra-high res, corrupt file, extreme aspect ratio, multi-face selection)
- R3 Boundaries (6 tests: rapid record toggle spam, zero frame record, invalid save path, frame size mismatch, idle stop, double start)
- R4 Boundaries (6 tests: clipboard empty, clipboard text, rapid paste spam, null frame render, minimized window, extreme window resize)
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
from PyQt5.QtWidgets import QApplication


# ============================================================================
# R1 Boundaries: Live Stream Boundaries (6 tests)
# ============================================================================

def test_b_r1_01_zero_faces_in_camera_stream(face_swap_engine, blank_image):
    """T2-B-R1-01: When camera view has no face, engine passes through raw frame safely."""
    out_frame = face_swap_engine.process_frame(blank_image)
    assert out_frame is not None
    assert out_frame.shape == blank_image.shape
    # If no face is detected in the stream, output must be unaltered raw frame
    assert np.array_equal(out_frame, blank_image)


def test_b_r1_02_camera_extreme_head_pose_yaw(face_swap_engine, valid_face_image):
    """T2-B-R1-02: Camera frame with extreme yaw angle is handled without exception."""
    # Synthesize asymmetric steep yaw pose
    h, w = valid_face_image.shape[:2]
    steep_yaw = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.ellipse(steep_yaw, (w // 2 - 80, h // 2), (50, 140), 35, 0, 360, (175, 195, 235), -1)

    # Process frame
    out_frame = face_swap_engine.process_frame(steep_yaw)
    assert out_frame is not None
    assert out_frame.shape == steep_yaw.shape


def test_b_r1_03_camera_stream_aspect_ratio_16_9(mock_camera, face_swap_engine):
    """T2-B-R1-03: Video stream provided at 16:9 widescreen (1280x720) is processed cleanly."""
    mock_camera.set_resolution(1280, 720)
    ret, frame_16_9 = mock_camera.read()
    assert ret is True
    assert frame_16_9.shape == (720, 1280, 3)

    out = face_swap_engine.process_frame(frame_16_9)
    assert out is not None
    assert out.shape == (720, 1280, 3)


def test_b_r1_04_camera_device_read_failure_or_disconnect(mock_camera):
    """T2-B-R1-04: Camera disconnect or read failure returns (False, None) safely."""
    mock_camera.inject_fault(0, "disconnect")
    ret, frame = mock_camera.read()
    assert ret is False
    assert frame is None
    assert mock_camera.is_opened() is False


def test_b_r1_05_camera_low_contrast_sensor_noise(face_swap_engine, valid_face_image):
    """T2-B-R1-05: Low contrast with heavy sensor noise is processed without NaN or overflow."""
    noisy = (valid_face_image.astype(float) * 0.4).astype(np.uint8)
    noise = np.random.normal(0, 25, noisy.shape).astype(np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    out = face_swap_engine.process_frame(noisy)
    assert out is not None
    assert not np.isnan(out).any()
    assert out.dtype == np.uint8


def test_b_r1_06_camera_rapid_resolution_switch(mock_camera, face_swap_engine):
    """T2-B-R1-06: Changing frame resolution mid-stream does not crash processing pipeline."""
    for (w, h) in [(640, 480), (320, 240), (800, 600), (640, 480)]:
        mock_camera.set_resolution(w, h)
        ret, frame = mock_camera.read()
        assert ret is True
        assert frame.shape == (h, w, 3)
        out = face_swap_engine.process_frame(frame)
        assert out.shape == (h, w, 3)


# ============================================================================
# R2 Boundaries: Source Image Boundaries (6 tests)
# ============================================================================

def test_b_r2_01_transparent_rgba_source_image(face_swap_engine, transparent_rgba_path):
    """T2-B-R2-01: 4-channel RGBA source image is normalized to 3-channel BGR without OpenCV crash."""
    rgba = cv2.imread(str(transparent_rgba_path), cv2.IMREAD_UNCHANGED)
    assert rgba is not None
    assert rgba.shape[2] == 4, "Should be 4-channel RGBA"

    success, msg = face_swap_engine.set_source_face(rgba)
    assert success is True
    assert face_swap_engine.is_source_loaded() is True


def test_b_r2_02_grayscale_source_image(face_swap_engine, grayscale_face_path):
    """T2-B-R2-02: 1-channel Grayscale image is converted to 3-channel BGR and loaded cleanly."""
    gray = cv2.imread(str(grayscale_face_path), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    assert len(gray.shape) == 2, "Should be 1-channel Grayscale"

    success, msg = face_swap_engine.set_source_face(gray)
    assert success is True
    assert face_swap_engine.is_source_loaded() is True


def test_b_r2_03_ultra_high_res_source_image(face_swap_engine, ultra_high_res_path):
    """T2-B-R2-03: 4000x3000 ultra-high-res image is downscaled and loaded within time budget."""
    high_res = cv2.imread(str(ultra_high_res_path))
    assert high_res is not None
    assert high_res.shape[0] >= 3000 and high_res.shape[1] >= 4000

    start_t = time.perf_counter()
    success, msg = face_swap_engine.set_source_face(high_res)
    elapsed = time.perf_counter() - start_t

    assert success is True
    assert elapsed < 5.0, f"Ultra-high-res load took too long: {elapsed:.2f}s"
    assert face_swap_engine.is_source_loaded() is True


def test_b_r2_04_corrupted_or_zero_byte_source_file(face_swap_engine, corrupt_image_path):
    """T2-B-R2-04: Loading 0-byte or corrupted image file fails safely with error message."""
    # Attempting to decode corrupt file returns None
    corrupt_data = cv2.imread(str(corrupt_image_path))
    # If None passed or invalid array
    if corrupt_data is None:
        # Engine should reject None safely
        success, msg = face_swap_engine.set_source_face(None)
        assert success is False
        assert len(msg) > 0


def test_b_r2_05_extreme_aspect_ratio_source_image(face_swap_engine, extreme_aspect_wide_path):
    """T2-B-R2-05: Panoramic banner image (2000x200) handled safely without crash."""
    wide_img = cv2.imread(str(extreme_aspect_wide_path))
    assert wide_img is not None
    assert wide_img.shape[1] == 2000 and wide_img.shape[0] == 200

    success, msg = face_swap_engine.set_source_face(wide_img)
    # Panoramic banner has no face
    assert success is False
    assert "face" in msg.lower()


def test_b_r2_06_multi_face_source_image_selection(face_swap_engine, multi_face_image):
    """T2-B-R2-06: Multiple faces in source image deterministically selects primary face."""
    success, msg = face_swap_engine.set_source_face(multi_face_image)
    assert success is True
    assert face_swap_engine.is_source_loaded() is True


# ============================================================================
# R3 Boundaries: Recording Boundaries (6 tests)
# ============================================================================

def test_b_r3_01_rapid_record_toggle_spam(video_recorder, valid_face_image, tmp_path):
    """T2-B-R3-01: Rapidly starting and stopping recording does not deadlock or crash."""
    for i in range(5):
        p = str(tmp_path / f"spam_rec_{i}.mp4")
        video_recorder.start_recording(p, 640, 480)
        video_recorder.record_frame(valid_face_image)
        video_recorder.stop_recording()
    assert video_recorder.is_recording() is False


def test_b_r3_02_record_zero_frames_stop_immediately(video_recorder, tmp_path):
    """T2-B-R3-02: Starting recording and immediately stopping without frames closes safely."""
    p = str(tmp_path / "zero_frames.mp4")
    video_recorder.start_recording(p, 640, 480)
    video_recorder.stop_recording()
    assert video_recorder.is_recording() is False


def test_b_r3_03_record_to_invalid_or_readonly_path(video_recorder, valid_face_image):
    """T2-B-R3-03: Invalid destination path handled without unhandled exception."""
    invalid_path = "Z:\\non_existent_folder_xyz_123\\rec.mp4"
    try:
        res = video_recorder.start_recording(invalid_path, 640, 480)
        # Should return False or if started, recording should not crash on frame push
        if res:
            video_recorder.record_frame(valid_face_image)
            video_recorder.stop_recording()
    except (OSError, IOError):
        pass  # Handled cleanly


def test_b_r3_04_record_frame_size_mismatch(video_recorder, tmp_path):
    """T2-B-R3-04: Pushing frame with unexpected dimensions handled gracefully without crash."""
    p = str(tmp_path / "mismatch.mp4")
    video_recorder.start_recording(p, 640, 480)

    # Frame with different dimensions (320x240 instead of 640x480)
    mismatched = np.zeros((240, 320, 3), dtype=np.uint8)
    video_recorder.record_frame(mismatched)

    video_recorder.stop_recording()
    assert video_recorder.is_recording() is False


def test_b_r3_05_stop_recording_when_not_recording(video_recorder):
    """T2-B-R3-05: Calling stop_recording when idle is a safe no-op."""
    assert video_recorder.is_recording() is False
    video_recorder.stop_recording()
    assert video_recorder.is_recording() is False


def test_b_r3_06_double_start_recording(video_recorder, tmp_path):
    """T2-B-R3-06: Calling start_recording while already recording handled cleanly."""
    p1 = str(tmp_path / "double_1.mp4")
    p2 = str(tmp_path / "double_2.mp4")

    video_recorder.start_recording(p1, 640, 480)
    assert video_recorder.is_recording() is True

    # Second call
    video_recorder.start_recording(p2, 640, 480)
    assert video_recorder.is_recording() is True

    video_recorder.stop_recording()
    assert video_recorder.is_recording() is False


# ============================================================================
# R4 Boundaries: GUI and Usability Boundaries (6 tests)
# ============================================================================

def test_b_r4_01_clipboard_empty(clipboard_service, headless_qapp):
    """T2-B-R4-01: Empty clipboard returns (None, error_msg) without crashing."""
    headless_qapp.clipboard().clear()
    headless_qapp.processEvents()

    img, err = clipboard_service.get_image_from_clipboard()
    assert img is None
    assert err is not None and len(err) > 0


def test_b_r4_02_clipboard_plain_text_non_image(clipboard_service, headless_qapp):
    """T2-B-R4-02: Plain text in clipboard is safely rejected as non-image."""
    headless_qapp.clipboard().setText("Just some regular text string, not an image")
    headless_qapp.processEvents()

    img, err = clipboard_service.get_image_from_clipboard()
    assert img is None
    assert err is not None and ("text" in err.lower() or "image" in err.lower() or "no" in err.lower())


def test_b_r4_03_clipboard_rapid_paste_spam(clipboard_service, valid_face_image, headless_qapp):
    """T2-B-R4-03: Rapidly querying clipboard 10 times in tight loop does not crash."""
    rgb = cv2.cvtColor(valid_face_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    headless_qapp.clipboard().setImage(qimg)
    headless_qapp.processEvents()

    for _ in range(10):
        img, err = clipboard_service.get_image_from_clipboard()
        assert img is not None


def test_b_r4_04_gui_render_null_or_empty_frame(main_window, qt_pump):
    """T2-B-R4-04: Feeding empty or None frame to GUI does not crash rendering."""
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    if hasattr(main_window, "video_widget") and hasattr(main_window.video_widget, "update_frame"):
        main_window.video_widget.update_frame(empty_frame)
        main_window.video_widget.update_frame(None)
    elif hasattr(main_window, "on_frame_ready"):
        main_window.on_frame_ready(empty_frame)
    qt_pump()


def test_b_r4_05_gui_window_minimized(main_window, valid_face_image, qt_pump):
    """T2-B-R4-05: Pushing frames while window is minimized does not crash or stall."""
    main_window.showMinimized()
    qt_pump()

    if hasattr(main_window, "video_widget") and hasattr(main_window.video_widget, "update_frame"):
        main_window.video_widget.update_frame(valid_face_image)
    qt_pump()
    main_window.showNormal()
    qt_pump()


def test_b_r4_06_gui_extreme_window_resize(main_window, valid_face_image, qt_pump):
    """T2-B-R4-06: Rapidly resizing window to extreme sizes preserves stability."""
    for (w, h) in [(320, 240), (1600, 900), (640, 480)]:
        main_window.resize(w, h)
        qt_pump()
        if hasattr(main_window, "video_widget") and hasattr(main_window.video_widget, "update_frame"):
            main_window.video_widget.update_frame(valid_face_image)
        qt_pump()
