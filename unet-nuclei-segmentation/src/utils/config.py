"""Pemuatan konfigurasi dan resolusi path proyek."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Ubah path relatif menjadi absolut terhadap root proyek."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Baca config YAML. Path default: configs/config.yaml."""
    config_path = resolve_path(path or "configs/config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Simpan config sebagai YAML."""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return output_path


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve semua entri pada bagian paths."""
    return {key: resolve_path(value) for key, value in config.get("paths", {}).items()}
