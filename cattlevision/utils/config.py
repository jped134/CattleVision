"""YAML configuration loader with dot-notation access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict subclass that supports attribute-style access (cfg.key)."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, Config):
                val = Config(val)
                self[name] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config file and optionally apply key=value overrides."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        _deep_update(data, overrides)
    return Config(data)


def _deep_update(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
