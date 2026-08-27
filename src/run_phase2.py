#!/usr/bin/env python
"""Run Phase 2 preprocessing and frozen train/validation in sequence.

Downloads are separate and restartable. This runner never opens held-out 2023
outcomes; final testing remains an explicit command.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, config: str, phase2: str, *extra: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / script),
                    "--config", config, "--phase2", phase2, *extra],
                   cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    if not args.skip_build:
        run("phase2_build.py", args.config, args.phase2, "--through", "validation")
    run("phase2_train.py", args.config, args.phase2)


if __name__ == "__main__":
    main()
