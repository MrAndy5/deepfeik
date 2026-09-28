"""Unit tests for IVideoSource, WebcamSource, and MockCameraSource."""
import time
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from deepfeik.pipeline.video_source import IVideoSource, WebcamSource
from deepfeik.pipeline.mock_camera import MockCameraSource, MockMode


# ============================================================================
# 1. IVideoSource Abstract Interface Tests
# ============================================================================

def test_ivideo_source_cannot_be_instantiated_directly():
    """IVideoSource is an abstract base class and cannot be instantiated."""
    with pytest.raises(TypeError) as excinfo:
        IVideoSource()
    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_ivideo_source_concrete_implementation():
    """Concrete subclass implementing all abstract methods can be instantiated."""
    class DummySource(IVideoSource):
        def open(self) -> bool:
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

        def is_opened(self) -> bool:
            return True

        def get_fps(self) -> float:
            return 30.0

        def get_resolution(self):
            return (640, 480)

    source = DummySource()
    assert isinstance(source, IVideoSource)
    assert source.is_opened() is True
    assert source.get_fps() == 30.0
    assert source.get_resolution() == (640, 480)


def test_ivideo_source_incomplete_subclass_rejected():
    """Subclass missing abstract methods cannot be instantiated."""
    class IncompleteSource(IVideoSource):
        def open(self) -> bool:
            return True

    with pytest.raises(TypeError):
        IncompleteSource()


# ============================================================================
# 2. WebcamSource Tests (Mocked Hardware)
# ============================================================================

@patch("cv2.VideoCapture")
def test_webcam_source_default_attributes(mock_cap_cls):
    """WebcamSource initializes with correct default properties."""
    source = WebcamSource(device_index=0, width=640, height=480, fps=30.0)
    assert source.get_resolution() == (640, 480)
    assert source.get_fps() == 30.0
    assert source.is_opened() is False


@patch("cv2.VideoCapture")
def test_webcam_source_open_success(mock_cap_cls):
    """WebcamSource.open() succeeds when hardware is accessible."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    success = source.open()

    assert success is True
    assert source.is_opened() is True
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)


@patch("cv2.VideoCapture")
def test_webcam_source_open_failure(mock_cap_cls):
    """WebcamSource.open() returns False when device cannot be opened."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=99)
    success = source.open()

    assert success is False
    assert source.is_opened() is False


@patch("cv2.VideoCapture")
def test_webcam_source_read_success(mock_cap_cls, sample_frame):
    """WebcamSource.read() returns (True, frame) when stream is active."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, sample_frame)
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    source.open()
    ret, frame = source.read()

    assert ret is True
    assert np.array_equal(frame, sample_frame)


@patch("cv2.VideoCapture")
def test_webcam_source_read_disconnect(mock_cap_cls):
    """WebcamSource.read() returns (False, None) on hardware disconnect."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    source.open()
    ret, frame = source.read()

    assert ret is False
    assert frame is None


def test_webcam_source_read_before_open():
    """Calling read() before open() returns (False, None)."""
    source = WebcamSource(device_index=0)
    ret, frame = source.read()
    assert ret is False
    assert frame is None


@patch("cv2.VideoCapture")
def test_webcam_source_directshow_fps_fallback(mock_cap_cls):
    """When DirectShow returns -1.0 for CAP_PROP_FPS, get_fps() falls back to 30.0."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = -1.0
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0, fps=30.0)
    source.open()
    assert source.get_fps() == 30.0


@patch("cv2.VideoCapture")
def test_webcam_source_release_lifecycle(mock_cap_cls):
    """WebcamSource.release() frees hardware resources and resets state."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    source.open()
    assert source.is_opened() is True

    source.release()
    assert source.is_opened() is False
    mock_cap.release.assert_called_once()
    assert source.read() == (False, None)


@patch("cv2.VideoCapture")
def test_webcam_source_idempotent_release(mock_cap_cls):
    """Calling release() multiple times does not raise exceptions."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    source.open()
    source.release()
    source.release()  # Second call must be safe
    assert source.is_opened() is False


@patch("cv2.VideoCapture")
def test_webcam_source_context_manager(mock_cap_cls, sample_frame):
    """WebcamSource can be used as a context manager."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, sample_frame)
    mock_cap_cls.return_value = mock_cap

    with WebcamSource(device_index=0) as source:
        assert source.is_opened() is True
        ret, frame = source.read()
        assert ret is True

    assert source.is_opened() is False
    mock_cap.release.assert_called_once()


# ============================================================================
# 3. MockCameraSource Tests (Synthetic Video Stream)
# ============================================================================

def test_mock_camera_open_and_is_opened():
    """MockCameraSource opens cleanly and reports correct status."""
    source = MockCameraSource(mode="synthetic", width=640, height=480, fps=30.0)
    assert source.is_opened() is False
    assert source.open() is True
    assert source.is_opened() is True
    source.release()
    assert source.is_opened() is False


def test_mock_camera_frame_shape_and_dtype():
    """MockCameraSource yields frames matching specified dimensions and BGR uint8 type."""
    source = MockCameraSource(mode="synthetic", width=640, height=480)
    source.open()
    ret, frame = source.read()
    source.release()

    assert ret is True
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_mock_camera_resolution_and_fps():
    """MockCameraSource returns configured resolution and FPS."""
    source = MockCameraSource(width=1280, height=720, fps=60.0)
    assert source.get_resolution() == (1280, 720)
    assert source.get_fps() == 60.0


def test_mock_camera_synthetic_procedural_motion():
    """Procedural synthetic mode produces dynamic motion across frames."""
    source = MockCameraSource(mode="synthetic", width=640, height=480)
    source.open()
    _, frame1 = source.read()
    _, frame2 = source.read()
    source.release()

    diff = np.abs(frame2.astype(np.int32) - frame1.astype(np.int32))
    assert np.max(diff) > 0, "Consecutive synthetic frames must not be static"


def test_mock_camera_static_image_mode(sample_frame):
    """Static image mode yields identical frames repeatedly."""
    source = MockCameraSource(mode="static", image=sample_frame, width=640, height=480)
    source.open()
    for _ in range(5):
        ret, frame = source.read()
        assert ret is True
        assert np.array_equal(frame, sample_frame)
    source.release()


def test_mock_camera_eof_behavior_without_loop(sample_frame):
    """When loop=False and frames are exhausted, read() returns (False, None)."""
    frames = [sample_frame.copy() for _ in range(3)]
    source = MockCameraSource(mode="sequence", frames=frames, loop=False)
    source.open()

    for _ in range(3):
        ret, frame = source.read()
        assert ret is True
        assert frame is not None

    ret, frame = source.read()
    assert ret is False
    assert frame is None
    source.release()


def test_mock_camera_loop_behavior(sample_frame):
    """When loop=True, frame sequence wraps around to beginning indefinitely."""
    frame_a = np.full((480, 640, 3), 10, dtype=np.uint8)
    frame_b = np.full((480, 640, 3), 200, dtype=np.uint8)
    source = MockCameraSource(mode="sequence", frames=[frame_a, frame_b], loop=True)
    source.open()

    # Read 4 frames (should cycle A -> B -> A -> B)
    _, f1 = source.read()
    _, f2 = source.read()
    _, f3 = source.read()
    _, f4 = source.read()
    source.release()

    assert np.array_equal(f1, frame_a)
    assert np.array_equal(f2, frame_b)
    assert np.array_equal(f3, frame_a)
    assert np.array_equal(f4, frame_b)


def test_mock_camera_scripted_failure_injection():
    """MockCameraSource simulates camera disconnection at specified frame."""
    source = MockCameraSource(mode="synthetic", failure_at_frame=3)
    source.open()

    ret1, f1 = source.read()  # frame 0
    ret2, f2 = source.read()  # frame 1
    ret3, f3 = source.read()  # frame 2
    ret4, f4 = source.read()  # frame 3 (failure injected)

    assert ret1 is True
    assert ret2 is True
    assert ret3 is True
    assert ret4 is False
    assert f4 is None
    assert source.is_opened() is False
    source.release()


def test_mock_camera_release_stops_read():
    """Calling release() causes subsequent read() calls to return (False, None)."""
    source = MockCameraSource(mode="synthetic")
    source.open()
    source.release()
    ret, frame = source.read()
    assert ret is False
    assert frame is None


def test_mock_camera_invalid_parameters():
    """MockCameraSource raises ValueError for invalid resolution or FPS."""
    with pytest.raises(ValueError):
        MockCameraSource(width=0, height=480)
    with pytest.raises(ValueError):
        MockCameraSource(width=640, height=-10)
    with pytest.raises(ValueError):
        MockCameraSource(fps=0.0)


def test_mock_camera_nonexistent_image_file():
    """MockCameraSource raises FileNotFoundError when image path does not exist."""
    with pytest.raises(FileNotFoundError):
        MockCameraSource(mode="static", image_path="non_existent_file_xyz.jpg")


@patch("cv2.VideoCapture")
def test_webcam_source_set_resolution_and_fps(mock_cap_cls):
    """WebcamSource.set_resolution() and set_fps() update properties."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    source.open()

    assert source.set_resolution(1280, 720) is True
    assert source.set_fps(60.0) is True


@patch("cv2.VideoCapture")
def test_webcam_source_reconnect(mock_cap_cls):
    """WebcamSource.reconnect() releases and re-opens the capture device."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0)
    source.open()
    assert source.is_opened() is True

    reconnected = source.reconnect()
    assert reconnected is True
    assert source.is_opened() is True


@patch("cv2.VideoCapture")
def test_webcam_source_auto_fallback_to_cap_any(mock_cap_cls):
    """WebcamSource falls back to CAP_ANY if initial backend fails to open."""
    failing_cap = MagicMock()
    failing_cap.isOpened.return_value = False

    working_cap = MagicMock()
    working_cap.isOpened.return_value = True

    mock_cap_cls.side_effect = [failing_cap, working_cap]

    source = WebcamSource(device_index=0, backend=cv2.CAP_DSHOW, auto_fallback=True)
    success = source.open()

    assert success is True
    assert source.is_opened() is True
    assert mock_cap_cls.call_count == 2


@patch("cv2.VideoCapture")
def test_webcam_source_consecutive_failures_disconnect(mock_cap_cls):
    """WebcamSource flags disconnected state after disconnect_threshold failures."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_cap_cls.return_value = mock_cap

    source = WebcamSource(device_index=0, disconnect_threshold=3)
    source.open()
    assert source.is_connected is True

    source.read()
    source.read()
    assert source.is_connected is True

    source.read()  # 3rd consecutive failure reaches threshold
    assert source.is_connected is False


def test_mock_camera_simulate_disconnect_reconnect():
    """MockCameraSource simulate_disconnect and simulate_reconnect control stream state."""
    source = MockCameraSource(mode="synthetic")
    source.open()
    assert source.is_opened() is True

    source.simulate_disconnect()
    assert source.is_opened() is False
    ret, frame = source.read()
    assert ret is False
    assert frame is None

    source.simulate_reconnect()
    assert source.is_opened() is True
    ret, frame = source.read()
    assert ret is True
    assert frame is not None


def test_mock_camera_set_resolution_and_fps():
    """MockCameraSource set_resolution and set_fps mutators."""
    source = MockCameraSource(mode="synthetic", width=640, height=480, fps=30.0)
    source.open()

    assert source.set_resolution(1280, 720) is True
    assert source.get_resolution() == (1280, 720)

    assert source.set_fps(60.0) is True
    assert source.get_fps() == 60.0

    # Invalid values return False
    assert source.set_resolution(-10, 480) is False
    assert source.set_fps(0.0) is False


def test_mock_camera_seek_and_frame_count():
    """MockCameraSource seek_frame and frame_count."""
    source = MockCameraSource(mode="synthetic")
    source.open()
    assert source.frame_count == 0

    source.read()
    assert source.frame_count == 1

    source.seek_frame(10)
    assert source.frame_count == 10


def test_mock_camera_cv2_get_set_duck_typing():
    """MockCameraSource supports cv2 VideoCapture get() and set() duck-typing."""
    source = MockCameraSource(mode="synthetic", width=640, height=480, fps=30.0)
    source.open()

    assert source.get(cv2.CAP_PROP_FRAME_WIDTH) == 640.0
    assert source.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480.0
    assert source.get(cv2.CAP_PROP_FPS) == 30.0
    assert source.get(cv2.CAP_PROP_POS_FRAMES) == 0.0

    assert source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0) is True
    assert source.get_resolution() == (1280, 480)

    assert source.set(cv2.CAP_PROP_FPS, 60.0) is True
    assert source.get_fps() == 60.0


def test_mock_camera_blank_and_noise_modes():
    """MockCameraSource supports pure BLANK and NOISE modes."""
    blank_source = MockCameraSource(mode="blank", width=640, height=480)
    blank_source.open()
    ret, frame = blank_source.read()
    assert ret is True
    assert np.all(frame == 0)

    noise_source = MockCameraSource(mode="noise", width=640, height=480)
    noise_source.open()
    ret, frame = noise_source.read()
    assert ret is True
    assert not np.all(frame == 0)


def test_mock_camera_black_and_noise_frame_injection():
    """MockCameraSource injects black and noise frames at specific indices."""
    source = MockCameraSource(mode="synthetic", black_frames={1}, noise_frames={2})
    source.open()

    _, f0 = source.read()  # normal
    assert not np.all(f0 == 0)

    _, f1 = source.read()  # black frame
    assert np.all(f1 == 0)

    _, f2 = source.read()  # noise frame
    assert not np.all(f2 == 0)


def test_mock_camera_freeze_at_frame():
    """MockCameraSource repeats identical frame once freeze_at_frame is reached."""
    source = MockCameraSource(mode="synthetic", freeze_at_frame=2)
    source.open()

    _, f0 = source.read()
    _, f1 = source.read()
    _, f2 = source.read()  # freeze triggers here
    _, f3 = source.read()  # must be identical to f2

    assert not np.array_equal(f0, f1)
    assert np.array_equal(f2, f3)


def test_mock_camera_timeline_script():
    """MockCameraSource executes multi-phase timeline script."""
    script = [
        {"start": 0, "end": 2, "type": "blank"},
        {"start": 2, "end": 4, "type": "dropout"},
    ]
    source = MockCameraSource(mode="scripted", timeline_script=script)
    source.open()

    ret0, f0 = source.read()
    assert ret0 is True
    assert np.all(f0 == 0)

    ret1, f1 = source.read()
    assert ret1 is True
    assert np.all(f1 == 0)

    ret2, f2 = source.read()  # dropout
    assert ret2 is False
    assert f2 is None


def test_mock_camera_expected_face_bbox_and_landmarks():
    """MockCameraSource returns expected analytical face bbox and landmark targets."""
    source = MockCameraSource(mode="synthetic", width=640, height=480)
    bbox = source.get_expected_face_bbox(frame_index=0)
    assert bbox is not None
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert w > 0 and h > 0

    lms = source.get_expected_landmarks(frame_index=0)
    assert lms is not None
    assert "left_eye" in lms
    assert "right_eye" in lms
    assert "nose_tip" in lms
    assert "mouth_center" in lms


def test_mock_camera_max_frames():
    """MockCameraSource terminates with (False, None) when max_frames is reached."""
    source = MockCameraSource(mode="synthetic", max_frames=2)
    source.open()

    assert source.read()[0] is True
    assert source.read()[0] is True
    assert source.read()[0] is False  # max_frames reached


def test_mock_camera_fail_duration_and_recovery():
    """MockCameraSource fail_at_frame with fail_duration recovers after duration."""
    source = MockCameraSource(mode="synthetic", fail_at_frame=2, fail_duration=2)
    source.open()

    assert source.read()[0] is True   # frame 0
    assert source.read()[0] is True   # frame 1
    assert source.read()[0] is False  # frame 2 (failed)
    assert source.read()[0] is False  # frame 3 (failed)
    assert source.read()[0] is True   # frame 4 (recovered!)


def test_mock_camera_realtime_pacing():
    """MockCameraSource realtime_pacing paces read calls."""
    source = MockCameraSource(mode="synthetic", fps=50.0, realtime_pacing=True)
    source.open()

    t0 = time.perf_counter()
    source.read()
    source.read()
    elapsed = time.perf_counter() - t0
    # At 50 FPS, interval is 0.02s
    assert elapsed >= 0.015
