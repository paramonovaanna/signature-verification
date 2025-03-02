import logging
import logging.config
from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json


def setup_logging(save_dir, log_config=None, default_level=logging.INFO, append=False):
    """
    Setup logging configuration.

    Args:
        save_dir (Path): path to directory, where all logs and
            checkpoints should be saved.
        log_config (str | None): path to logger config. If none
            'logger_config.json' from the src.logger directory is used.
        default_level (int): default logging level.
        append (bool): if True, append file instead of overwriting.
    """
    log_config = Path(ROOT_PATH / "src" / "logger" / "logger_config.json")
    config = read_json(log_config)
    # modify logging paths based on run config
    for _, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(save_dir / handler["filename"])

    logging.config.dictConfig(config)
