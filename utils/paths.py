from __future__ import annotations
from pathlib import Path
from datetime import datetime

import logging
logger = logging.getLogger("growthsim")

def ensure_output_dir(output: str | None, simulation: str) -> str:
    """
    Ensures an output directory exists.
    - If output is None/empty: creates ./outputs/<timestamp>_<simulation>/
    - If output is provided: creates it if missing
    
    Args:
        output (str | None): Desired output directory path. If None or empty, a timestamp

    Returns:
        str: Path to the ensured output directory.
    """
    sim = (simulation or "SIM").strip().upper()

    if output is None or str(output).strip() == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"{stamp}_{sim}/"
        logger.info("No output directory specified, using '%s'", out_dir)

    else:
        out_dir = Path(output)

    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir.resolve())
