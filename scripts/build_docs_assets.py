"""Regenerate every figure used in the README / docs and the project-overview PDF.

    python scripts/build_docs_assets.py

1. runs the default configuration into a temporary folder,
2. copies figures 1-6, the GIF and the dashboard to ``docs/images``,
3. builds ``docs/project_overview.pdf``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from tool_wear_demo import WearLoop, load_config

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        config = load_config(ROOT / "config" / "default.yaml", {"output": {"directory": tmp}})
        loop = WearLoop(config, Path(tmp) / "docs_run")
        result = loop.run()
        figs = loop.run_dir / "figures"
        for f in figs.iterdir():
            target = ROOT / "docs" / "dashboard.html" if f.name == "dashboard.html" else IMAGES / f.name
            shutil.copy2(f, target)
        shutil.copy2(loop.run_dir / "cycle_001" / "geometry_after.png", IMAGES / "cycle_001_geometry_after.png")
        shutil.copy2(loop.run_dir / f"cycle_{result.cycles_completed:03d}" / "process_fields.png", IMAGES / "final_cycle_process_fields.png")
        shutil.copy2(loop.run_dir / "history.csv", IMAGES / "history_default_run.csv")
        print(f"copied figures to {IMAGES}")

    from build_project_overview import build  # noqa: E402  (same folder)

    build(ROOT / "docs" / "project_overview.pdf", IMAGES)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
