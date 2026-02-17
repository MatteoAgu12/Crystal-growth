from __future__ import annotations
from pathlib import Path

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

def setup_logging(verbose: bool = False, log_file: str | None = None) -> logging.Logger:
    """
    Global Logger configuration for the entire app.
     - verbose=False -> INFO
     - verbose=True  -> DEBUG
     - log_file -> writes on file
    """
    level = logging.DEBUG if verbose else logging.INFO

    # reset handler (utile se rilanci da notebook/IDE)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger("growthsim")
