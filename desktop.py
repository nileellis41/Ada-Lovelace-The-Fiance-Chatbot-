"""Ada Lovelace — Finance Research AI desktop application."""
from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Ada Lovelace")
    app.setOrganizationName("Finance Research AI")

    window = MainWindow()
    window.show()

    logger.info("Ada Lovelace started")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
