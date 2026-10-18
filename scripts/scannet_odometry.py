"""Run selectable RGB-D odometry on ScanNet using the shared runner."""

from scene_recall.odometry.cli import main
from scene_recall.odometry.io import save_run

__all__ = ["main", "save_run"]

if __name__ == "__main__":
    main()
