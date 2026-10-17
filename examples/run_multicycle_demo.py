"""Run the full multi-cycle demo and create all figures, the GIF and the dashboard.

    python examples/run_multicycle_demo.py                 # default configuration (~30 s)
    python examples/run_multicycle_demo.py --fast          # coarse mesh, a few seconds
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tool_wear_demo import WearLoop, load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="use config/demo_fast.yaml")
    args = parser.parse_args()

    config_file = ROOT / "config" / ("demo_fast.yaml" if args.fast else "default.yaml")
    config = load_config(config_file, {"output": {"directory": str(ROOT / "results")}})
    loop = WearLoop(config)
    print(f"configuration: {config_file.name}\nrun folder:    {loop.run_dir}\n")
    result = loop.run()

    print(f"\nstop reason:               {result.stop_reason}")
    print(f"cycles completed:          {result.cycles_completed}")
    print(f"flank-wear indicator:      {result.final_flank_wear_mm * 1000:.1f} um")
    print(f"crater depth indicator:    {result.history[-1]['crater_depth_indicator_um']:.1f} um")
    print(f"figures + dashboard:       {loop.run_dir / 'figures'}")


if __name__ == "__main__":
    main()
