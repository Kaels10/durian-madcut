"""
Auto-load a YOLO model on startup so kiosk / Pi users can open the app and detect immediately.

Resolution order (first existing file that loads wins):
  1. Environment variable DURIAN_MODEL (full path to .onnx or .pt)
  2. Last successful path stored in model_autoload.json (project root)
  3. models/best.onnx, models/model.onnx, best.onnx in project root
  4. models/best.pt, models/model.pt (PC / PyTorch installs)

After any successful load (here or from Settings), the path is saved for next launch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.ml.detector import DurianDetector

CONFIG_FILENAME = "model_autoload.json"


def project_root() -> Path:
    """Repository root (directory that contains main.py)."""
    return Path(__file__).resolve().parent.parent.parent


def config_path() -> Path:
    return project_root() / CONFIG_FILENAME


def read_last_saved_path() -> str | None:
    p = config_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        s = data.get("model_path")
        if isinstance(s, str) and s.strip():
            return s.strip()
    except (OSError, json.JSONDecodeError):
        pass
    return None


def persist_last_model_path(path: str) -> None:
    """Remember a working model path for the next app launch."""
    try:
        resolved = str(Path(path).expanduser().resolve(strict=False))
        config_path().write_text(
            json.dumps({"model_path": resolved}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[model_autoload] Could not save {CONFIG_FILENAME}: {exc}")


def _dedupe_preserve_order(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        try:
            key = str(p.resolve(strict=False))
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def iter_autoload_candidates() -> list[Path]:
    root = project_root()
    raw: list[Path] = []
    env = os.environ.get("DURIAN_MODEL", "").strip()
    if env:
        raw.append(Path(env))
    last = read_last_saved_path()
    if last:
        raw.append(Path(last))
    for name in ("best.onnx", "model.onnx"):
        raw.append(root / "models" / name)
    raw.append(root / "best.onnx")
    for name in ("best.pt", "model.pt"):
        raw.append(root / "models" / name)
    return _dedupe_preserve_order(raw)


def try_autoload_on_startup(detector: DurianDetector) -> str | None:
    """
    Try each candidate until load_model succeeds.
    Returns the path loaded, or None if no model could be loaded.
    """
    for path in iter_autoload_candidates():
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf not in (".onnx", ".pt"):
            continue
        if detector.load_model(str(path)):
            persist_last_model_path(str(path))
            return str(path)
    return None
