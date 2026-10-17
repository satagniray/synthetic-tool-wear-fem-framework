"""(Re)create figures 1-6, the progression GIF and the HTML dashboard for a run.

    python examples/plot_results.py                       # most recent run (results/LATEST)
    python examples/plot_results.py results/run_20261011_120000
"""

from __future__ import annotations

import sys
from pathlib import Path

from tool_wear_demo.visualization import make_run_outputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "latest"
    for name, path in make_run_outputs(target).items():
        print(f"{name:22s} {path}")


if __name__ == "__main__":
    main()
