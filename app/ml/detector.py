"""
app/ml/detector.py
Durian maturity detection.

- **ONNX (.onnx)** — ONNX Runtime on CPU (recommended for Raspberry Pi 5; no PyTorch).
- **PyTorch (.pt)** — Ultralytics YOLO when `ultralytics` + `torch` are installed.

Classes: mature · immature · damaged
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.utils.pi import is_raspberry_pi

# Class colour map (BGR for OpenCV, RGB for PIL)
CLASS_COLORS = {
    "mature":   {"bgr": (34, 197, 94),  "rgb": (34, 197, 94),  "hex": "#22C55E"},
    "immature": {"bgr": (234, 179, 8),  "rgb": (234, 179, 8),  "hex": "#EAB308"},
    "damaged":  {"bgr": (239, 68, 68),  "rgb": (239, 68, 68),  "hex": "#EF4444"},
}
DEFAULT_COLOR = {"bgr": (148, 163, 184), "rgb": (148, 163, 184), "hex": "#94A3B8"}

# When ONNX metadata has no names, assume Roboflow / project order for 3-class models.
_FALLBACK_THREE_CLASS = ["mature", "immature", "damaged"]

# Map training export names to app vocabulary (colours + UI).
_LABEL_ALIASES = {
    "defective": "damaged",
}


def normalize_maturity_label(label: str) -> str:
    """Lowercase + spaces→underscores; map synonyms (e.g. defective → damaged)."""
    s = str(label).strip().lower().replace(" ", "_")
    return _LABEL_ALIASES.get(s, s)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _letterbox(
    image: np.ndarray,
    new_size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with padding to a square *new_size*; returns image, ratio r, (dw, dh)."""
    h, w = image.shape[:2]
    r = min(new_size / h, new_size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_size - nw) // 2, (new_size - nh) // 2
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    out[dh : dh + nh, dw : dw + nw] = resized
    return out, r, (dw, dh)


def _unletterbox_xyxy(
    xyxy: np.ndarray,
    r: float,
    pad: tuple[int, int],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Map boxes from letterboxed coords to original image size."""
    dw, dh = pad
    out = xyxy.copy().astype(np.float32)
    out[:, [0, 2]] -= dw
    out[:, [1, 3]] -= dh
    out /= r
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h - 1)
    return out.astype(np.int32)


class DurianDetector:
    """ONNX Runtime or Ultralytics YOLO backend for durian maturity detection."""

    def __init__(self):
        self.model = None  # Ultralytics YOLO when using .pt
        self._session = None  # onnxruntime.InferenceSession when using .onnx
        self._input_name: str = ""
        self._onnx_out_shapes: list[Any] = []

        self.model_path: Optional[str] = None
        self.backend: str = ""  # "onnx" | "torch"
        self.conf_threshold: float = 0.5
        self.iou_threshold: float = 0.45
        # Default to 640 so common ONNX exports load without manual settings.
        self.imgsz: int = 640
        self.device: str = "cpu"
        self._class_names: list[str] = []

    # ------------------------------------------------------------------
    @property
    def runtime_label(self) -> str:
        if self.backend == "onnx":
            return f"ONNX · imgsz={self.imgsz}"
        if self.backend == "torch":
            return f"PyTorch ({self.device.upper()}) · imgsz={self.imgsz}"
        return "—"

    def is_loaded(self) -> bool:
        return self.backend == "onnx" and self._session is not None or (
            self.backend == "torch" and self.model is not None
        )

    # ------------------------------------------------------------------
    def load_model(self, path: str) -> bool:
        path = str(path)
        ext = Path(path).suffix.lower()
        self._unload()
        self.model_path = path

        if ext == ".onnx":
            ok = self._load_onnx(path)
            if not ok:
                self.model_path = None
            return ok
        if ext == ".pt":
            ok = self._load_torch(path)
            if not ok:
                self.model_path = None
            return ok

        print(f"[Detector] Unsupported model type: {ext} (use .onnx on Pi, or .pt with PyTorch)")
        self.model_path = None
        return False

    def _unload(self) -> None:
        self.model = None
        self._session = None
        self._input_name = ""
        self._onnx_out_shapes = []
        self.backend = ""
        self._class_names = []

    # ------------------------------------------------------------------
    def _load_onnx(self, path: str) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[Detector] onnxruntime not installed. On Pi: pip install onnxruntime")
            return False

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Pi: avoid saturating every CPU with ORT so capture + Tk keep cycles.
            # Model input stays imgsz×imgsz (default 640); this only changes threading.
            if is_raspberry_pi():
                try:
                    cores = max(1, (os.cpu_count() or 4))
                    opts.intra_op_num_threads = max(1, cores - 1)
                    opts.inter_op_num_threads = 1
                except Exception:
                    pass
            self._session = ort.InferenceSession(
                path,
                opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._onnx_out_shapes = [o.shape for o in self._session.get_outputs()]
            self._class_names = self._onnx_class_names()
            self.backend = "onnx"

            dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
            self._session.run(None, {self._input_name: dummy})
            return True
        except Exception as exc:
            print(f"[Detector] Failed to load ONNX: {exc}")
            self._unload()
            return False

    def _onnx_class_names(self) -> list[str]:
        try:
            meta = self._session.get_modelmeta().custom_metadata_map
            raw = meta.get("names")
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    keys = sorted(data.keys(), key=lambda k: int(k))
                    return [str(data[k]) for k in keys]
                if isinstance(data, list):
                    return [str(x) for x in data]
        except Exception:
            pass
        return []

    def _load_torch(self, path: str) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            print(
                "[Detector] ultralytics/torch not installed — use a .onnx model on Pi, "
                "or install: pip install -r requirements-torch.txt"
            )
            return False

        try:
            self.model = YOLO(path)
            self._class_names = list(self.model.names.values())
            self.backend = "torch"
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            return True
        except Exception as exc:
            print(f"[Detector] Failed to load PyTorch model: {exc}")
            self._unload()
            return False

    def _resolve_class_names(self, nc: int) -> list[str]:
        if len(self._class_names) == nc:
            return self._class_names
        if nc == len(_FALLBACK_THREE_CLASS):
            return list(_FALLBACK_THREE_CLASS)
        return [f"class_{i}" for i in range(nc)]

    # ------------------------------------------------------------------
    def predict_pil(self, pil_image: Image.Image) -> list[dict]:
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Open Settings and load a .onnx or .pt file.")

        rgb = np.array(pil_image.convert("RGB"))
        if self.backend == "onnx":
            return self._predict_onnx(rgb)
        return self._predict_torch_array(rgb)

    def predict(self, image_path: str) -> list[dict]:
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Open Settings and load a .onnx or .pt file.")

        if self.backend == "onnx":
            rgb = np.array(Image.open(image_path).convert("RGB"))
            return self._predict_onnx(rgb)

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._boxes_from_ultralytics(results)

    def _predict_torch_array(self, rgb: np.ndarray) -> list[dict]:
        results = self.model.predict(
            source=rgb,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._boxes_from_ultralytics(results)

    def _boxes_from_ultralytics(self, results) -> list[dict]:
        detections: list[dict] = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            cls_id = int(box.cls[0].item())
            raw = self.model.names.get(cls_id, str(cls_id))
            label = normalize_maturity_label(str(raw))
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({"label": label, "confidence": conf, "bbox": (x1, y1, x2, y2)})
        return detections

    def _predict_onnx(self, rgb: np.ndarray) -> list[dict]:
        orig_h, orig_w = rgb.shape[:2]
        lb, r, pad = _letterbox(rgb, self.imgsz)
        blob = lb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outs = self._session.run(None, {self._input_name: blob})
        dets = self._parse_onnx_outputs(outs, orig_w, orig_h, r, pad)
        return dets

    def _parse_onnx_outputs(
        self,
        outs: list[np.ndarray],
        orig_w: int,
        orig_h: int,
        r: float,
        pad: tuple[int, int],
    ) -> list[dict]:
        """Parse Ultralytics-style raw head or embedded-NMS ONNX outputs."""
        out0 = outs[0]
        if out0.dtype == np.float16:
            out0 = out0.astype(np.float32)

        # Embedded NMS: [1, N, 6] or [N, 6]  → x1,y1,x2,y2,score,cls
        if out0.ndim >= 2 and out0.shape[-1] == 6:
            pred = out0.reshape(-1, 6)
            pred = pred[np.any(pred[:, :4] != 0, axis=1)]
            boxes = pred[:, :4].astype(np.float32)
            scores = pred[:, 4]
            cls_ids = pred[:, 5].astype(np.int32)
            keep = scores >= self.conf_threshold
            boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
            if len(boxes) == 0:
                return []
            boxes = _unletterbox_xyxy(boxes, r, pad, orig_w, orig_h)
            names = self._resolve_class_names(int(cls_ids.max()) + 1 if len(cls_ids) else 3)
            out: list[dict] = []
            for i in range(len(boxes)):
                cid = int(cls_ids[i])
                raw = names[cid] if cid < len(names) else str(cid)
                label = normalize_maturity_label(str(raw))
                x1, y1, x2, y2 = boxes[i].tolist()
                out.append(
                    {
                        "label": label,
                        "confidence": float(scores[i]),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    }
                )
            return out

        # Raw YOLO head: [1, 4+nc, A] or [1, A, 4+nc]
        pred = np.squeeze(out0)
        if pred.ndim != 2:
            return []

        if pred.shape[0] < pred.shape[1]:  # (4+nc, A)
            pred = pred.T
        # pred: (A, 4+nc)
        boxes_c = pred[:, :4].copy()
        cls_scores = pred[:, 4:]

        if cls_scores.size == 0:
            return []

        if cls_scores.max() > 1.25 or cls_scores.min() < -0.25:
            cls_scores = _sigmoid(cls_scores)

        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        keep = scores >= self.conf_threshold
        boxes_c = boxes_c[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        if len(boxes_c) == 0:
            return []

        # cx, cy, w, h → xyxy in letterboxed image space
        cx, cy, w, h = boxes_c[:, 0], boxes_c[:, 1], boxes_c[:, 2], boxes_c[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        xyxy = _unletterbox_xyxy(xyxy, r, pad, orig_w, orig_h)

        # NMS (xywh for OpenCV)
        wh = xyxy[:, 2:4] - xyxy[:, 0:2]
        nms_in = np.concatenate([xyxy[:, :2], wh], axis=1)
        idxs = cv2.dnn.NMSBoxes(
            nms_in.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if idxs is None or len(idxs) == 0:
            return []
        idxs = np.array(idxs).reshape(-1)

        nc = int(cls_scores.shape[1])
        names = self._resolve_class_names(nc)

        detections: list[dict] = []
        for i in idxs:
            cid = int(cls_ids[i])
            raw = names[cid] if cid < len(names) else str(cid)
            label = normalize_maturity_label(str(raw))
            x1, y1, x2, y2 = xyxy[i].tolist()
            detections.append(
                {
                    "label": label,
                    "confidence": float(scores[i]),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return detections

    # ------------------------------------------------------------------
    def draw_boxes(self, pil_image: Image.Image, detections: list[dict]) -> Image.Image:
        """
        Draw bounding boxes with labels placed beside each box (prefer right side).
        """
        img = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        box_thick = max(2, int(min(w, h) * 0.003))

        font_size = max(14, int(min(w, h) * 0.032))
        font = self._get_font(font_size)

        for det in detections:
            label = normalize_maturity_label(det["label"])
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(label, DEFAULT_COLOR)["rgb"]

            for thickness in range(box_thick):
                draw.rectangle(
                    [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                    outline=color,
                )

            text = label.replace("_", " ").capitalize()
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except AttributeError:
                tw, th = draw.textsize(text, font=font)

            pad = 5
            gap = 6
            box_h = th + pad * 2

            # Prefer label to the right of the box, vertically centred on the box
            ty = int(y1 + (y2 - y1) / 2 - box_h / 2)
            ty = max(2, min(ty, h - box_h - 2))

            tx = x2 + gap
            if tx + tw + pad * 2 > w - 2:
                tx = x1 - tw - pad * 2 - gap
            if tx < 2:
                tx = min(max(2, x1), w - tw - pad * 2 - 2)
                ty = max(0, y1 - box_h - gap)
                if ty < 2:
                    ty = min(y2 + gap, h - box_h - 2)

            # Label panel
            draw.rectangle(
                [tx, ty, tx + tw + pad * 2, ty + box_h],
                fill=color,
                outline=color,
            )
            draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

        return img

    # ------------------------------------------------------------------
    def summarise(self, detections: list[dict]) -> dict:
        summary = {cls: {"count": 0, "conf_sum": 0.0} for cls in CLASS_COLORS}
        for det in detections:
            lbl = normalize_maturity_label(det["label"])
            if lbl in summary:
                summary[lbl]["count"] += 1
                summary[lbl]["conf_sum"] += det["confidence"]
        result = {}
        for cls, data in summary.items():
            count = data["count"]
            avg_conf = (data["conf_sum"] / count) if count else 0.0
            result[cls] = {"count": count, "avg_confidence": avg_conf}
        return result

    @staticmethod
    def _get_font(size: int):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            try:
                return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
            except Exception:
                return ImageFont.load_default()
