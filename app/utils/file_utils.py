"""
app/utils/file_utils.py
Helpers for scanning image files from folders.
"""

from __future__ import annotations
import os
from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_images(folder: str) -> list[str]:
    """Return sorted list of image file paths inside *folder* (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def list_images_recursive(folder: str) -> list[str]:
    """Return sorted list of image file paths inside *folder* and all subdirs."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        str(p) for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_pil_image(path: str) -> Image.Image:
    """Load and return a PIL Image, converting to RGB."""
    return Image.open(path).convert("RGB")


def get_file_size_str(path: str) -> str:
    """Return human-readable file size string."""
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
