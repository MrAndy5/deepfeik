"""
deepfeik.__main__

Entry point: python -m deepfeik [--camera N]
"""

import argparse
import logging
import sys

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
    args = parser.parse_args()

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
