"""
deepfeik.__main__

Entry point: python -m deepfeik [--camera N]
"""

import argparse
import logging
import os
import sys

# Ensure stdout and stderr exist even when packaged as a Windows GUI executable
if sys.platform == "win32":
    try:
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):
            if sys.stdout is None or getattr(sys.stdout, "closed", False):
                sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            if sys.stderr is None or getattr(sys.stderr, "closed", False):
                sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
    except Exception:
        pass

if sys.stdout is None:
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass
if sys.stderr is None:
    try:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass

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
        print("deepfeik: verification successful")
        return 0

    _configure_logging(args.verbose)

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
