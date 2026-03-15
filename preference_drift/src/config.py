"""Configuration loading with YAML inheritance support."""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load config with optional base inheritance via _base_ key."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "_base_" in cfg:
        base_path = (Path(config_path).parent / cfg.pop("_base_")).resolve()
        base_cfg = load_config(str(base_path))
        cfg = deep_merge(base_cfg, cfg)

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
