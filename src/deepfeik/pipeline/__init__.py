"""
deepfeik.pipeline

Video capture, buffering, recording, and pipeline coordination subsystem.
"""

from deepfeik.pipeline.video_source import IVideoSource, WebcamSource
from deepfeik.pipeline.mock_camera import MockCameraSource, MockMode
from deepfeik.pipeline.frame_buffer import LatestFrameBuffer, BufferStats, BufferClosedError

__all__ = [
    "IVideoSource",
    "WebcamSource",
    "MockCameraSource",
    "MockMode",
    "LatestFrameBuffer",
    "BufferStats",
    "BufferClosedError",
]
