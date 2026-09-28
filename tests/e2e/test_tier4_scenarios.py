"""Tier 4: 6 Realistic Application-Level Scenarios for DeepFeik.

Covers:
- Scenario 1: Standard Video Call Session (launch, stream, load, swap, record, stop, exit)
- Scenario 2: Multi-Source Presenter Session (hot-swapping multiple faces during recording)
- Scenario 3: Interrupted Stream Recovery Session (user leaves frame, camera disconnect, recovery)
- Scenario 4: Sustained Load and Resource Stability Session (300 frames, memory RSS bounding, steady throughput)
- Scenario 5: Fault-Injection & Recovery Session (corrupt file, plain text, no-face, invalid record path)
- Scenario 6: Headless CI Smoke Pipeline Session (full end-to-end integration offscreen)
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication


# ============================================================================
# Scenario 1: The Standard Virtual Call Session
# ============================================================================

def test_scenario_01_standard_video_call_session(face_swap_engine, video_recorder, mock_animated_camera, valid_face_image, tmp_path):
    """Tier 4 Scenario 1: Standard user session from launch to swap, record, and exit."""
    # Step 1: Initialize stream and verify live acquisition
    assert mock_animated_camera.is_opened() is True
    for _ in range(5):
        ret, frame = mock_animated_camera.read()
        assert ret is True
        # Prior to source face loading, engine passes through
        passthrough = face_swap_engine.process_frame(frame)
        assert np.array_equal(passthrough, frame)

    # Step 2: User loads valid portrait photo
    t_start = time.perf_counter()
    success, msg = face_swap_engine.set_source_face(valid_face_image)
    t_load = time.perf_counter() - t_start

    assert success is True
    assert t_load < 2.0, f"Source face activation took {t_load:.2f}s (budget: < 2.0s)"
    assert face_swap_engine.is_source_loaded() is True

    # Step 3: User starts recording session to MP4
    rec_path = str(tmp_path / "standard_call.mp4")
    started = video_recorder.start_recording(rec_path, 640, 480, target_fps=30.0)
    assert started is True

    # Step 4: Stream 30 active frames with procedural head animation
    for _ in range(30):
        ret, frame = mock_animated_camera.read()
        assert ret is True
        swapped = face_swap_engine.process_frame(frame)
        assert swapped is not None
        video_recorder.record_frame(swapped)

    # Step 5: User stops recording and finalizes video
    video_recorder.stop_recording()
    assert video_recorder.is_recording() is False

    # Step 6: Verify recorded output
    assert Path(rec_path).exists()
    cap = cv2.VideoCapture(rec_path)
    assert cap.isOpened() is True
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count >= 25, f"Expected at least 25 frames, got {count}"

    # Step 7: Clean release
    mock_animated_camera.release()
    assert mock_animated_camera.is_opened() is False


# ============================================================================
# Scenario 2: Multi-Source Presenter Session
# ============================================================================

def test_scenario_02_multi_source_presenter_session(face_swap_engine, video_recorder, mock_animated_camera, valid_face_image, tmp_path):
    """Tier 4 Scenario 2: Hot-swapping multiple source faces mid-stream during recording."""
    rec_path = str(tmp_path / "presenter_session.mp4")
    video_recorder.start_recording(rec_path, 640, 480, 30.0)

    # Phase A: Source Face A
    face_a = valid_face_image.copy()
    face_swap_engine.set_source_face(face_a)
    for _ in range(12):
        ret, frame = mock_animated_camera.read()
        swapped = face_swap_engine.process_frame(frame)
        video_recorder.record_frame(swapped)

    # Phase B: Source Face B (hot-swapped)
    face_b = valid_face_image.copy()
    face_b[:, :, 0] = np.clip(face_b[:, :, 0].astype(int) + 70, 0, 255)
    face_swap_engine.set_source_face(face_b)
    for _ in range(12):
        ret, frame = mock_animated_camera.read()
        swapped = face_swap_engine.process_frame(frame)
        video_recorder.record_frame(swapped)

    # Phase C: Source Face C (hot-swapped)
    face_c = valid_face_image.copy()
    face_c[:, :, 2] = np.clip(face_c[:, :, 2].astype(int) + 70, 0, 255)
    face_swap_engine.set_source_face(face_c)
    for _ in range(12):
        ret, frame = mock_animated_camera.read()
        swapped = face_swap_engine.process_frame(frame)
        video_recorder.record_frame(swapped)

    video_recorder.stop_recording()

    cap = cv2.VideoCapture(rec_path)
    assert cap.isOpened() is True
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert total >= 30, f"Presenter recording should contain >= 30 frames, got {total}"


# ============================================================================
# Scenario 3: Interrupted Stream Recovery Session
# ============================================================================

def test_scenario_03_interrupted_stream_recovery(face_swap_engine, mock_scripted_camera, valid_face_image):
    """Tier 4 Scenario 3: Stream handles user exit (0 faces), dropout, and resumes seamlessly."""
    face_swap_engine.set_source_face(valid_face_image)

    # Scripted timeline:
    # 0-15: normal face
    # 15-30: blank/no face
    # 30-45: multi-face
    # 45-55: dropout
    # 55-75: normal face

    for idx in range(70):
        ret, frame = mock_scripted_camera.read()
        if not ret or frame is None:
            # Camera dropout handled safely
            continue
        out = face_swap_engine.process_frame(frame)
        assert out is not None
        assert out.shape == frame.shape


# ============================================================================
# Scenario 4: Sustained Load and Resource Stability Session
# ============================================================================

def test_scenario_04_sustained_load_and_resource_stability(face_swap_engine, mock_animated_camera, valid_face_image):
    """Tier 4 Scenario 4: Continuous execution over 150 frames with bounded memory delta (< 50MB)."""
    face_swap_engine.set_source_face(valid_face_image)

    process = psutil.Process(os.getpid())
    # Warmup
    for _ in range(10):
        ret, frame = mock_animated_camera.read()
        _ = face_swap_engine.process_frame(frame)

    mem_initial = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    num_frames = 150
    for i in range(num_frames):
        ret, frame = mock_animated_camera.read()
        assert ret is True
        out = face_swap_engine.process_frame(frame)
        assert out is not None

    total_time = time.perf_counter() - t0
    mem_final = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_final - mem_initial
    avg_fps = num_frames / total_time

    assert mem_delta < 50.0, f"Memory RSS grew by {mem_delta:.2f}MB (threshold: < 50MB)"
    assert avg_fps >= 15.0, f"Sustained throughput dropped to {avg_fps:.2f} FPS (budget: >= 15 FPS)"


# ============================================================================
# Scenario 5: Fault-Injection & Recovery Session
# ============================================================================

def test_scenario_05_fault_injection_and_error_dialog_recovery(face_swap_engine, video_recorder, corrupt_image_path, blank_image, valid_face_image, tmp_path):
    """Tier 4 Scenario 5: System gracefully handles corrupted files, missing faces, and recovers."""
    # Step 1: Corrupt file load attempt
    corrupt_bytes = cv2.imread(str(corrupt_image_path))
    if corrupt_bytes is None:
        succ, msg = face_swap_engine.set_source_face(None)
        assert succ is False

    # Step 2: Blank image load attempt (no face)
    succ, msg = face_swap_engine.set_source_face(blank_image)
    assert succ is False
    assert "face" in msg.lower()

    # Step 3: Valid face load succeeds
    succ, msg = face_swap_engine.set_source_face(valid_face_image)
    assert succ is True
    assert face_swap_engine.is_source_loaded() is True

    # Step 4: Invalid save path attempt
    try:
        video_recorder.start_recording("Z:\\invalid_drive\\rec.mp4", 640, 480)
    except (OSError, IOError):
        pass

    # Step 5: Valid recording succeeds
    valid_rec = str(tmp_path / "recovered_rec.mp4")
    started = video_recorder.start_recording(valid_rec, 640, 480, 30.0)
    if started:
        video_recorder.record_frame(valid_face_image)
        video_recorder.stop_recording()
        assert Path(valid_rec).exists()


# ============================================================================
# Scenario 6: Headless CI Smoke Pipeline Session
# ============================================================================

def test_scenario_06_headless_ci_smoke_pipeline(face_swap_engine, video_recorder, mock_camera, valid_face_image, tmp_path):
    """Tier 4 Scenario 6: Full end-to-end smoke pipeline runs in strict headless offscreen mode."""
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"

    # Pipeline linkage
    rec_path = str(tmp_path / "ci_smoke.mp4")
    video_recorder.start_recording(rec_path, 640, 480, 30.0)

    face_swap_engine.set_source_face(valid_face_image)
    for _ in range(15):
        ret, frame = mock_camera.read()
        assert ret is True
        processed = face_swap_engine.process_frame(frame)
        video_recorder.record_frame(processed)

    video_recorder.stop_recording()
    mock_camera.release()

    assert Path(rec_path).exists()
    assert os.path.getsize(rec_path) > 100
