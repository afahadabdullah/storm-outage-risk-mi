"""Config loading and path management.

region.yaml is the base; phase1.yaml overrides it key-by-key. Nothing in the
pipeline reads a region-specific constant from anywhere else.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data" / "raw"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    models: Path = ROOT / "models"
    figures: Path = ROOT / "figures"
    logs: Path = ROOT / "logs"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed,
                  self.models, self.figures, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths().ensure()


class Config(dict):
    """dict with attribute access and a required() accessor that fails loudly."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def required(self, key: str) -> Any:
        if key not in self or self[key] in (None, "", "AUTO"):
            raise KeyError(
                f"config key {key!r} is unset or still 'AUTO'. "
                f"Run `make window` first if this is window_start/window_end."
            )
        return self[key]

    @property
    def is_phase1(self) -> bool:
        return int(self.get("phase", 0)) == 1


def load_config(region: str | Path, phase1: str | Path | None = None) -> Config:
    cfg = Config(yaml.safe_load(Path(region).read_text()))
    cfg["_region_file"] = str(region)
    if phase1:
        overrides = yaml.safe_load(Path(phase1).read_text()) or {}
        cfg["_phase1_overrides"] = sorted(overrides)
        cfg["_phase1_file"] = str(phase1)
        cfg.update(overrides)
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    p.add_argument("--phase1", default=str(ROOT / "config" / "phase1.yaml"),
                   help="phase 1 override file; pass '' to run without overrides")
    p.add_argument("--synthetic", action="store_true",
                   help="generate stand-in data instead of downloading")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.phase1 or None)


def state_prefixes(cfg: Config) -> set[str]:
    return {str(s).zfill(2) for s in cfg["state_fips"]}
