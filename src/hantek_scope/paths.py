from __future__ import annotations
import os
from pathlib import Path


def get_repo_root() -> Path:
    # paths.py лежит в src/hantek_scope/
    return Path(__file__).resolve().parents[2]


def get_default_sdk_dir() -> Path:
    return get_repo_root() / "Hantek_SDK"


def get_default_dll_dir() -> Path:
    env = os.environ.get("HANTEK_DLL_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"HANTEK_DLL_DIR does not exist: {p}")
        return p

    p = get_default_sdk_dir() / "Dll" / "x64"
    if not p.exists():
        raise FileNotFoundError(
            "Default DLL directory not found. "
            "Set HANTEK_DLL_DIR or place SDK at Hantek_SDK/Dll/x64"
        )
    return p