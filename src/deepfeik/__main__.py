"""
deepfeik.__main__

Entry point: python -m deepfeik [--camera N]
"""

import argparse
import logging
import os
import sys

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from deepfeik.gui.main_window import MainWindow


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
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
        import mediapipe
        import deepfeik.core.engine
        sys.stdout.write("deepfeik: verification successful\n")
        sys.stdout.flush()
        return 0

    _configure_logging(args.verbose)

    # Hide background console window on Windows unless verbose mode was requested
    if sys.platform == "win32" and not args.verbose:
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass

    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("deepfeik")
    app.setOrganizationName("deepfeik")

    window = MainWindow(device_index=args.camera)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
