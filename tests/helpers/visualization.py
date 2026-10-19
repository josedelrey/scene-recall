"""Run the recording CLI from the same environment as the optional SDK."""

import subprocess
import sys


def rrd(command, path, *extra):
    result = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", command, str(path), *extra],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return result.stdout
