import logging
import sys
from typing import Optional


def init_root_logger(filepath: Optional[str] = None, stdout=True):
    """
    Initialise the root logger.

    Example of use after this function :
    > import logging
    > logging.info('Marvelous!')

    Parameters
    ----------
    filepath : str, optional
        The path of the log file. Logs are appended to this file.
    stdout : bool
        If true, the logger sends the logs to stdout. Otherwise, it does not.
    """
    logger = logging.getLogger()

    # Remove default handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    if filepath is not None:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
