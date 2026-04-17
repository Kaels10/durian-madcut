"""
app/utils/export.py
Export detection results to CSV.
"""

from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path


def export_csv(results: list[dict], filepath: str) -> int:
    """
    Write *results* to a CSV file at *filepath*.

    Each result dict should have keys:
        filename, label, confidence, bbox, timestamp

    Returns the number of rows written.
    """
    fieldnames = ["filename", "label", "confidence_%", "bbox_x1", "bbox_y1",
                  "bbox_x2", "bbox_y2", "timestamp"]

    rows = []
    for r in results:
        bbox = r.get("bbox", (0, 0, 0, 0))
        rows.append({
            "filename":     Path(r.get("filename", "")).name,
            "label":        r.get("label", ""),
            "confidence_%": f"{r.get('confidence', 0) * 100:.2f}",
            "bbox_x1":      bbox[0] if len(bbox) > 0 else "",
            "bbox_y1":      bbox[1] if len(bbox) > 1 else "",
            "bbox_x2":      bbox[2] if len(bbox) > 2 else "",
            "bbox_y2":      bbox[3] if len(bbox) > 3 else "",
            "timestamp":    r.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def generate_csv_filename(base_dir: str = ".") -> str:
    """Return a timestamped CSV filename inside *base_dir*."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(base_dir) / f"durian_results_{ts}.csv")
