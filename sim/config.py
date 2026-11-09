"""Configuration: nested dicts loaded from YAML, merged over defaults.

``configs/default.yaml`` holds every knob with comments; experiment configs
only override what they change. Dotted overrides (``router.policy=early``)
are applied on top, which is what sweeps use.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_overrides(items: Iterable[str]) -> dict:
    out: dict = {}
    for item in items:
        k, _, v = item.partition("=")
        set_dotted(out, k.strip(), yaml.safe_load(v))
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT) as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT.resolve():
        with open(path) as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
