"""
app/ml/detector.py
YOLOv11 wrapper for durian maturity detection.
Classes: mature, immature, damaged
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Class colour map (BGR for OpenCV, RGB for PIL)
CLASS_COLORS = {
    "mature":   {"bgr": (34, 197, 94),  "rgb": (34, 197, 94),  "hex": "#22C55E"},
    "immature": {"bgr": (234, 179, 8),  "rgb": (234, 179, 8),  "hex": "#EAB308"},
    "damaged":  {"bgr": (239, 68, 68),  "rgb": (239, 68, 68),  "hex": "#EF4444"},
}
DEFAULT_COLOR = {"bgr": (148, 163, 184), "rgb": (148, 163, 184), "hex": "#94A3B8"}


class DurianDetector:
    """Wraps an Ultralytics YOLOv11 model for durian maturity detection."""

    def __init__(self):
        self.model = None
        self.model_path: Optional[str] = None
        self.conf_threshold: float = 0.5
        self.imgsz: int = 640
        self.device: str = "cpu"
        self._class_names: list[str] = []

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def load_model(self, path: str) -> bool:
        """Load a YOLO .pt model from *path*. Returns True on success."""
        try:
            from ultralytics import YOLO  # lazy import so app opens without ultralytics
            self.model = YOLO(path)
            self.model_path = path
            self._class_names = list(self.model.names.values())
            # Warm-up on a blank image
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, conf=self.conf_threshold,
                               imgsz=self.imgsz, device=self.device, verbose=False)
            return True
        except Exception as exc:
            print(f"[Detector] Failed to load model: {exc}")
            self.model = None
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_pil(self, pil_image: Image.Image) -> list[dict]:
        """
        Run inference directly on a PIL Image object (e.g. from a camera frame).
        Returns the same detection-dict list as :meth:`predict`.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a .pt file in Settings.")

        import numpy as np
        arr = np.array(pil_image.convert("RGB"))
        results = self.model.predict(
            source=arr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections = []
        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.model.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    })
        return detections

    def predict(self, image_path: str) -> list[dict]:
        """
        Run inference on *image_path*.

        Returns a list of detection dicts:
            {label, confidence, bbox: (x1,y1,x2,y2)}
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a .pt file in Settings.")

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.model.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    })
        return detections

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def draw_boxes(self, pil_image: Image.Image, detections: list[dict]) -> Image.Image:
        """
        Draw bounding boxes with labels over *pil_image*.
        Returns a new PIL Image (does not modify original).
        """
        img = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        box_thick = max(2, int(min(w, h) * 0.003))

        # Try to load a font, fall back to default
        font_size = max(14, int(min(w, h) * 0.035))
        font = self._get_font(font_size)
        small_font = self._get_font(max(11, font_size - 4))

        for det in detections:
            label = det["label"].lower()
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(label, DEFAULT_COLOR)["rgb"]

            # Bounding box
            for thickness in range(box_thick):
                draw.rectangle(
                    [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                    outline=color,
                )

            # Label background
            text = f"{label.capitalize()}  {conf * 100:.1f}%"
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
            pad = 4
            label_y = max(0, y1 - th - pad * 2)
            draw.rectangle(
                [x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2],
                fill=color,
            )
            draw.text((x1 + pad, label_y + pad), text, fill=(255, 255, 255), font=font)

        return img

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------
    def summarise(self, detections: list[dict]) -> dict:
        """Return per-class counts and average confidence."""
        summary = {cls: {"count": 0, "conf_sum": 0.0} for cls in CLASS_COLORS}
        for det in detections:
            lbl = det["label"].lower()
            if lbl in summary:
                summary[lbl]["count"] += 1
                summary[lbl]["conf_sum"] += det["confidence"]
        result = {}
        for cls, data in summary.items():
            count = data["count"]
            avg_conf = (data["conf_sum"] / count) if count else 0.0
            result[cls] = {"count": count, "avg_confidence": avg_conf}
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_font(size: int):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            try:
                return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
            except Exception:
                return ImageFont.load_default()
