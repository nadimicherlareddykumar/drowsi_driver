import logging
import os
import sys

from dms.config import MODEL_PATH
from dms.ui.app import run_app


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


if __name__ == "__main__":
    configure_logging()
    logger = logging.getLogger("dri01")
    if not os.path.exists(MODEL_PATH):
        logger.critical("%s missing.", MODEL_PATH)
        sys.exit(1)
    run_app()
