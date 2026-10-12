"""
deepfeik.__main__

Entry point: python -m deepfeik [--camera N]
"""

import argparse
import datetime
import logging
import os
from pathlib import Path
import sys
import traceback

# Setup robust crash log path across platforms
_LOG_DIR = Path.home() / ".deepfeik"
_CRASH_LOG = _LOG_DIR / "crash.log"
_APP_LOG = _LOG_DIR / "deepfeik.log"


def _ensure_log_dir() -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _show_fatal_dialog(title: str, message: str) -> None:
    """Displays a native UI dialog even if Qt has not initialized or failed."""
    if sys.platform == "win32":
        try:
            import ctypes
            # MB_ICONERROR (0x10) | MB_OK (0x0) | MB_SETFOREGROUND (0x10000)
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10 | 0x0 | 0x10000)
            return
        except Exception:
            pass

    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        created = False
        if app is None:
            app = QApplication(sys.argv)
            created = True
        QMessageBox.critical(None, title, message)
        if created:
            app.quit()
    except Exception:
        pass


def _unhandled_exception_hook(exc_type, exc_value, exc_tb) -> None:
    """Global exception hook: logs full traceback to disk and shows user dialogue."""
    _ensure_log_dir()
    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    
    timestamp = datetime.datetime.now().isoformat()
    header = f"\n{'='*70}\n[FATAL ERROR] {timestamp}\n{'='*70}\n"
    full_entry = header + err_msg

    try:
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(full_entry)
        with open(_APP_LOG, "a", encoding="utf-8") as f:
            f.write(full_entry)
    except Exception:
        pass

    sys.stderr.write(full_entry)
    sys.stderr.flush()

    user_feedback = (
        f"deepfeik encountered a fatal error and must close.\n\n"
        f"Error: {exc_value}\n\n"
        f"Log file written to:\n{_CRASH_LOG}\n\n"
        f"Please share this log file to report this issue."
    )
    _show_fatal_dialog("deepfeik — Fatal Error", user_feedback)


sys.excepthook = _unhandled_exception_hook

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


def _configure_logging(verbose: bool = False) -> None:
    _ensure_log_dir()
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(str(_APP_LOG), mode="a", encoding="utf-8"))
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="deepfeik",
        description="Real-time face swap using your webcam.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="N",
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        metavar="W",
        help="Webcam capture width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        metavar="H",
        help="Webcam capture height in pixels (default: 480)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--check-install",
        action="store_true",
        help="Verify dependencies and core models load properly and exit immediately.",
    )
    args = parser.parse_args()

    if args.check_install:
        import cv2
        import numpy
        import PIL
        import deepfeik.core.engine
        sys.stdout.write("deepfeik: verification successful\n")
        sys.stdout.flush()
        return 0

    _configure_logging(args.verbose)
    logging.info("Starting deepfeik... log file: %s", _APP_LOG)

    # Initialize early GPU environment hints (Optimus/OpenCL/dGPU)
    try:
        from deepfeik.core.gpu import setup_gpu_environment_hints, configure_gpu_acceleration
        setup_gpu_environment_hints()
        configure_gpu_acceleration()
    except Exception as e:
        logging.warning("GPU acceleration initialization note: %s", e)

    # Note: On Windows, we keep console attached until Qt successfully initializes
    # so startup diagnostics are never silently discarded.
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from deepfeik.gui.main_window import MainWindow

    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("deepfeik")
    app.setOrganizationName("deepfeik")

    logging.info("Initializing main window...")
    window = MainWindow(device_index=args.camera, width=args.width, height=args.height)
    window.show()

    logging.info("deepfeik application window launched successfully.")
    exit_code = app.exec_()
    logging.info("Application exited with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    exit_val = 0
    try:
        exit_val = main()
    except SystemExit as se:
        exit_val = se.code if se.code is not None else 0
    except Exception as exc:
        _unhandled_exception_hook(*sys.exc_info())
        exit_val = 1

    if exit_val != 0:
        sys.stderr.write(f"\n[deepfeik] Exited with error code {exit_val}. Log saved to: {_APP_LOG}\n")
        sys.stderr.flush()

    sys.exit(exit_val)
