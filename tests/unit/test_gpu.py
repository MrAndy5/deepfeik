"""
tests.unit.test_gpu

Unit tests for deepfeik.core.gpu auto-detection and hardware acceleration configuration.
Verifies fallback logic, mock environments (no OpenCL, failed device, dGPU detection),
and summary formatting.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest

from deepfeik.core.gpu import (
    configure_gpu_acceleration,
    get_gpu_status_summary,
    setup_gpu_environment_hints,
)


def test_setup_gpu_environment_hints(monkeypatch):
    """Ensures environment variables for OpenCV OpenCL dGPU preference are set."""
    monkeypatch.delenv("OPENCV_OPENCL_DEVICE", raising=False)
    setup_gpu_environment_hints()
    import os
    assert os.environ.get("OPENCV_OPENCL_DEVICE") == ":DGPU"
    assert os.environ.get("OPENCV_OPENCL_RAISE_ERROR") == "0"


def test_configure_gpu_acceleration_on_current_machine():
    """Validates real detection on host environment (doesn't crash and returns required keys)."""
    info = configure_gpu_acceleration(force_refresh=True)
    assert isinstance(info, dict)
    assert "accelerated" in info
    assert "device_name" in info
    assert "vendor" in info
    assert "is_dgpu" in info
    assert "opencl_enabled" in info
    assert isinstance(info["accelerated"], bool)
    assert isinstance(info["is_dgpu"], bool)


def test_fallback_when_opencl_not_available():
    """Simulates an environment where cv2.ocl.haveOpenCL() returns False."""
    mock_cv2 = MagicMock()
    mock_cv2.ocl.haveOpenCL.return_value = False

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        info = configure_gpu_acceleration(force_refresh=True)
        assert info["accelerated"] is False
        assert info["device_name"] == "CPU"
        assert info["is_dgpu"] is False
        assert info["opencl_enabled"] is False


def test_fallback_when_set_use_opencl_fails():
    """Simulates OpenCL present but useOpenCL returns False after setUseOpenCL(True)."""
    mock_cv2 = MagicMock()
    mock_cv2.ocl.haveOpenCL.return_value = True
    mock_cv2.ocl.useOpenCL.return_value = False

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        info = configure_gpu_acceleration(force_refresh=True)
        assert info["accelerated"] is False
        assert info["device_name"] == "CPU"
        assert info["is_dgpu"] is False


def test_dgpu_detected_discrete():
    """Simulates a discrete NVIDIA or AMD GPU with hostUnifiedMemory == False."""
    mock_cv2 = MagicMock()
    mock_cv2.ocl.haveOpenCL.return_value = True
    mock_cv2.ocl.useOpenCL.return_value = True

    mock_dev = MagicMock()
    mock_dev.available.return_value = True
    mock_dev.name.return_value = "NVIDIA GeForce RTX 4080"
    mock_dev.vendorName.return_value = "NVIDIA Corporation"
    mock_dev.hostUnifiedMemory.return_value = False
    mock_dev.isNVidia.return_value = True
    mock_dev.maxComputeUnits.return_value = 76
    mock_dev.globalMemSize.return_value = 16 * 1024 * 1024 * 1024

    mock_cv2.ocl.Device.getDefault.return_value = mock_dev

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        info = configure_gpu_acceleration(force_refresh=True)
        assert info["accelerated"] is True
        assert info["is_dgpu"] is True
        assert "GeForce" in info["device_name"]
        assert info["compute_units"] == 76
        assert info["memory_mb"] > 10000

        summary = get_gpu_status_summary()
        assert "(dGPU)" in summary
        assert "NVIDIA GeForce RTX 4080" in summary


def test_igpu_detected_integrated():
    """Simulates an integrated Intel UHD GPU with hostUnifiedMemory == True."""
    mock_cv2 = MagicMock()
    mock_cv2.ocl.haveOpenCL.return_value = True
    mock_cv2.ocl.useOpenCL.return_value = True

    mock_dev = MagicMock()
    mock_dev.available.return_value = True
    mock_dev.name.return_value = "Intel(R) UHD Graphics 630"
    mock_dev.vendorName.return_value = "Intel Corporation"
    mock_dev.hostUnifiedMemory.return_value = True
    mock_dev.isNVidia.return_value = False
    mock_dev.maxComputeUnits.return_value = 24
    mock_dev.globalMemSize.return_value = 1024 * 1024 * 1024

    mock_cv2.ocl.Device.getDefault.return_value = mock_dev

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        info = configure_gpu_acceleration(force_refresh=True)
        assert info["accelerated"] is True
        assert info["is_dgpu"] is False
        assert "Intel" in info["device_name"]

        summary = get_gpu_status_summary()
        assert "GPU: Intel(R) UHD Graphics 630" in summary
        assert "(dGPU)" not in summary
