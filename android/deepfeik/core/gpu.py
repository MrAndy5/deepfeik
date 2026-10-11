"""
deepfeik.core.gpu

Hardware acceleration and dedicated GPU auto-detection module.

Provides automatic detection and configuration of dedicated GPUs (dGPUs) such as
NVIDIA GeForce/RTX, AMD Radeon RX/Discrete, or Intel Arc.
Configures OpenCV OpenCL execution, environment hints for hybrid graphics systems
(NVIDIA Optimus / AMD PowerXpress), and returns hardware telemetry safely without
crashing on CPU-only systems.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cached accelerator information
_CACHED_GPU_INFO: Optional[Dict[str, Any]] = None


def setup_gpu_environment_hints() -> None:
    """Configures process environment variables and Windows exports to prefer dGPU.

    Should be called as early as possible during process startup before OpenCV
    initializes device contexts.
    """
    try:
        # Instruct OpenCV OpenCL backend to prioritize dedicated GPUs over integrated ones
        if "OPENCV_OPENCL_DEVICE" not in os.environ:
            os.environ["OPENCV_OPENCL_DEVICE"] = ":DGPU"

        # Prevent OpenCL compilation stalls from hanging UI thread indefinitely
        if "OPENCV_OPENCL_RAISE_ERROR" not in os.environ:
            os.environ["OPENCV_OPENCL_RAISE_ERROR"] = "0"

        # On Windows hybrid laptops (Optimus / PowerXpress), request high performance
        if sys.platform == "win32":
            # Set NVIDIA CUDA visible devices hint if unconfigured
            if "CUDA_DEVICE_ORDER" not in os.environ:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed setting GPU environment hints: %s", exc)


def configure_gpu_acceleration(force_refresh: bool = False) -> Dict[str, Any]:
    """Detects and activates dedicated GPU acceleration for OpenCV operations.

    Zero-configuration:
      1. Sets hybrid graphics environment hints to prioritize discrete GPU.
      2. Initializes OpenCV OpenCL and evaluates whether a dedicated GPU is available.
      3. Verifies device is not a host-unified memory device (i.e. discrete VRAM) or
         matches known discrete vendors (NVIDIA / AMD / Intel Arc).
      4. Falls back cleanly to CPU if no dGPU is present or OpenCL is unavailable.

    Args:
        force_refresh: If True, re-detects even if previously cached.

    Returns:
        dict containing:
            - 'accelerated' (bool): True if dGPU is active and enabled.
            - 'device_name' (str): Name of the active GPU or 'CPU'.
            - 'vendor' (str): GPU vendor (e.g., 'NVIDIA Corporation') or 'CPU'.
            - 'is_dgpu' (bool): True if a discrete GPU was detected and selected.
            - 'opencl_enabled' (bool): True if OpenCV OpenCL is currently active.
            - 'memory_mb' (float): Total device memory in MB, or 0.0.
            - 'compute_units' (int): Compute unit count, or 0.
    """
    global _CACHED_GPU_INFO
    if _CACHED_GPU_INFO is not None and not force_refresh:
        return _CACHED_GPU_INFO

    # Apply process-level environment hints first
    setup_gpu_environment_hints()

    info: Dict[str, Any] = {
        "accelerated": False,
        "device_name": "CPU",
        "vendor": "CPU",
        "is_dgpu": False,
        "opencl_enabled": False,
        "memory_mb": 0.0,
        "compute_units": 0,
    }

    try:
        import cv2

        # Optimize OpenCV multithreaded primitives
        cv2.setUseOptimized(True)

        # Check OpenCL support in OpenCV binary
        if not cv2.ocl.haveOpenCL():
            logger.info("OpenCV OpenCL is not supported on this platform. Running on CPU.")
            _CACHED_GPU_INFO = info
            return info

        # Enable OpenCL in OpenCV
        cv2.ocl.setUseOpenCL(True)

        if not cv2.ocl.useOpenCL():
            logger.info("Unable to enable OpenCL. Running on CPU.")
            _CACHED_GPU_INFO = info
            return info

        dev = cv2.ocl.Device.getDefault()
        if not dev or not dev.available():
            logger.info("No active OpenCL device found. Running on CPU.")
            cv2.ocl.setUseOpenCL(False)
            _CACHED_GPU_INFO = info
            return info

        dev_name = dev.name() or "Unknown GPU"
        vendor = dev.vendorName() or "Unknown Vendor"
        compute_units = dev.maxComputeUnits()
        mem_mb = round(dev.globalMemSize() / (1024.0 * 1024.0), 1)

        # Check if device is a discrete/dedicated GPU
        # In OpenCL: hostUnifiedMemory() is False on discrete GPUs (they have dedicated VRAM).
        # On iGPUs (Intel UHD, AMD Vega APU), hostUnifiedMemory() is True.
        # Alternatively, dev.isNVidia() almost always indicates a discrete GPU on laptops/desktops.
        is_discrete = False
        try:
            is_discrete = not dev.hostUnifiedMemory()
        except Exception:
            pass

        if not is_discrete and (dev.isNVidia() or "geforce" in dev_name.lower() or "rtx" in dev_name.lower()):
            is_discrete = True

        info["opencl_enabled"] = True
        info["device_name"] = dev_name
        info["vendor"] = vendor
        info["memory_mb"] = mem_mb
        info["compute_units"] = compute_units
        info["is_dgpu"] = is_discrete
        info["accelerated"] = True

        if is_discrete:
            logger.info(
                "Dedicated GPU detected and activated: %s (%s, %.0f MB VRAM, %d CUs)",
                dev_name, vendor, mem_mb, compute_units
            )
        else:
            logger.info(
                "GPU acceleration active on: %s (%s, %.0f MB shared)",
                dev_name, vendor, mem_mb
            )

    except Exception as exc:
        logger.warning("GPU detection/configuration encountered an error: %s. Using CPU fallback.", exc)
        info["accelerated"] = False
        info["is_dgpu"] = False
        info["opencl_enabled"] = False
        try:
            import cv2
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

    _CACHED_GPU_INFO = info
    return info


def get_gpu_status_summary() -> str:
    """Returns a short human-readable string suitable for UI status bars or logs.

    Examples:
      - "GPU: NVIDIA GeForce GTX 1650 Ti (dGPU)"
      - "GPU: Intel Iris Xe"
      - "CPU mode"
    """
    info = configure_gpu_acceleration()
    if info.get("accelerated"):
        dev = info.get("device_name", "GPU")
        if info.get("is_dgpu"):
            return f"GPU: {dev} (dGPU)"
        return f"GPU: {dev}"
    return "CPU"
