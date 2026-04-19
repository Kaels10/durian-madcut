"""Raspberry Pi detection helpers (used for sane defaults)."""

from __future__ import annotations


def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"raspberry" in f.read().lower()
    except OSError:
        return False
