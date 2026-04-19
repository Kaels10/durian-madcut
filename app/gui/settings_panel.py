"""
app/gui/settings_panel.py
Settings panel: model path, confidence threshold, device, image size.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from app.ml.detector import DurianDetector
from app.gui.theme import COLORS, FONTS
from app.utils.model_autoload import persist_last_model_path


class SettingsPanel(tk.Frame):
    def __init__(self, parent, detector: DurianDetector, on_model_loaded=None, **kwargs):
        super().__init__(parent, bg=COLORS["bg"], **kwargs)
        self.detector = detector
        self.on_model_loaded = on_model_loaded

        self._model_path_var = tk.StringVar(value=detector.model_path or "")
        self._conf_var = tk.DoubleVar(value=detector.conf_threshold)
        self._imgsz_var = tk.IntVar(value=detector.imgsz)
        self._device_var = tk.StringVar(value=detector.device)

        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        # ---- Title ----
        tk.Label(self, text="⚙  Settings", font=FONTS["h1"],
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", padx=32, pady=(32, 4))
        tk.Label(
            self,
            text=(
                "On first setup, copy your ONNX to models/best.onnx (Pi) — it loads automatically next time. "
                "Or browse below. PyTorch .pt works on a PC with GPU."
            ),
            font=FONTS["body"], bg=COLORS["bg"], fg=COLORS["muted"],
            wraplength=720, justify="left",
        ).pack(anchor="w", padx=32)

        content = tk.Frame(self, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=32, pady=24)

        # ---- Model card ----
        self._model_card(content)
        # ---- Inference card ----
        self._inference_card(content)
        # ---- About card ----
        self._about_card(content)

    # ------------------------------------------------------------------
    def _card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=COLORS["card"], bd=0)
        outer.pack(fill="x", pady=10)
        tk.Label(outer, text=title, font=FONTS["h2"],
                 bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 8))
        sep = tk.Frame(outer, bg=COLORS["border"], height=1)
        sep.pack(fill="x", padx=20, pady=(0, 12))
        body = tk.Frame(outer, bg=COLORS["card"])
        body.pack(fill="x", padx=20, pady=(0, 16))
        return body

    # ------------------------------------------------------------------
    def _model_card(self, parent):
        body = self._card(parent, "🤖  Model")

        # Path row
        path_row = tk.Frame(body, bg=COLORS["card"])
        path_row.pack(fill="x", pady=4)

        tk.Label(path_row, text="Model (.onnx / .pt)", font=FONTS["label"],
                 bg=COLORS["card"], fg=COLORS["muted"], width=18, anchor="w").pack(side="left")

        entry = tk.Entry(path_row, textvariable=self._model_path_var,
                         font=FONTS["body"], bg=COLORS["input"], fg=COLORS["text"],
                         insertbackground=COLORS["text"], relief="flat",
                         highlightthickness=1, highlightcolor=COLORS["accent"],
                         highlightbackground=COLORS["border"])
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

        browse_btn = tk.Button(path_row, text="Browse",
                               font=FONTS["body"], bg=COLORS["accent"], fg="white",
                               activebackground=COLORS["accent_hover"], activeforeground="white",
                               relief="flat", padx=16, pady=6, cursor="hand2",
                               command=self._browse_model)
        browse_btn.pack(side="left")

        # Status row
        self._model_status_var = tk.StringVar(value=self._model_status_text())
        self._model_status_label = tk.Label(body, textvariable=self._model_status_var,
                                            font=FONTS["small"], bg=COLORS["card"],
                                            fg=COLORS["success"] if self.detector.is_loaded() else COLORS["danger"])
        self._model_status_label.pack(anchor="w", pady=(4, 0))

        # Load button
        load_row = tk.Frame(body, bg=COLORS["card"])
        load_row.pack(fill="x", pady=(12, 0))
        tk.Button(load_row, text="  Load Model  ",
                  font=FONTS["h2"], bg=COLORS["success"], fg="white",
                  activebackground=COLORS["success_hover"], activeforeground="white",
                  relief="flat", padx=20, pady=8, cursor="hand2",
                  command=self._load_model).pack(side="left")

    def _inference_card(self, parent):
        body = self._card(parent, "🔧  Inference Options")

        rows = [
            ("Confidence Threshold", self._conf_var, 0.1, 0.95, 0.05, "float"),
            ("Image Size (px)",      self._imgsz_var, 320, 1280, 32,  "int"),
        ]
        for label, var, from_, to, resolution, kind in rows:
            row = tk.Frame(body, bg=COLORS["card"])
            row.pack(fill="x", pady=6)
            tk.Label(row, text=label, font=FONTS["label"],
                     bg=COLORS["card"], fg=COLORS["muted"], width=22, anchor="w").pack(side="left")
            disp = tk.Label(row, text=str(var.get()), font=FONTS["body"],
                            bg=COLORS["card"], fg=COLORS["accent"], width=6)
            disp.pack(side="right")

            def _update(val, v=var, d=disp, k=kind):
                d.config(text=f"{float(val):.2f}" if k == "float" else str(int(float(val))))

            scale = ttk.Scale(row, variable=var, from_=from_, to=to,
                              orient="horizontal", command=_update)
            scale.pack(side="left", fill="x", expand=True, padx=8)

        # Device
        dev_row = tk.Frame(body, bg=COLORS["card"])
        dev_row.pack(fill="x", pady=6)
        tk.Label(dev_row, text="Device", font=FONTS["label"],
                 bg=COLORS["card"], fg=COLORS["muted"], width=22, anchor="w").pack(side="left")
        for dev in ("cpu", "cuda", "mps"):
            rb = tk.Radiobutton(dev_row, text=dev.upper(), variable=self._device_var, value=dev,
                                font=FONTS["body"], bg=COLORS["card"], fg=COLORS["text"],
                                selectcolor=COLORS["bg"], activebackground=COLORS["card"],
                                relief="flat")
            rb.pack(side="left", padx=8)

        # Apply
        tk.Button(body, text="Apply",
                  font=FONTS["body"], bg=COLORS["accent"], fg="white",
                  activebackground=COLORS["accent_hover"], activeforeground="white",
                  relief="flat", padx=16, pady=6, cursor="hand2",
                  command=self._apply_settings).pack(anchor="e", pady=(12, 0))

    def _about_card(self, parent):
        body = self._card(parent, "ℹ️  About")
        lines = [
            ("Application", "Durian Maturity Assessment v1.0"),
            ("Model Type",  "YOLOv11 (Ultralytics)"),
            ("Classes",     "Mature · Immature · Damaged"),
            ("Platform",    "Raspberry Pi / Windows"),
        ]
        for key, val in lines:
            row = tk.Frame(body, bg=COLORS["card"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=key, font=FONTS["label"], bg=COLORS["card"],
                     fg=COLORS["muted"], width=16, anchor="w").pack(side="left")
            tk.Label(row, text=val, font=FONTS["body"], bg=COLORS["card"],
                     fg=COLORS["text"]).pack(side="left")

    # ------------------------------------------------------------------
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model",
            filetypes=[
                ("ONNX (Pi)", "*.onnx"),
                ("PyTorch", "*.pt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._model_path_var.set(path)

    def _load_model(self):
        path = self._model_path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please browse and select a .onnx or .pt model file.")
            return
        if not Path(path).exists():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        # Give visual feedback
        self._model_status_var.set("⏳ Loading model…")
        self._model_status_label.config(fg=COLORS["muted"])
        self.update_idletasks()

        ok = self.detector.load_model(path)
        if ok:
            persist_last_model_path(path)
            self._model_status_var.set(f"✅ Model loaded: {Path(path).name}")
            self._model_status_label.config(fg=COLORS["success"])
            if self.on_model_loaded:
                self.on_model_loaded(path)
        else:
            self._model_status_var.set("❌ Failed to load model. Check console for details.")
            self._model_status_label.config(fg=COLORS["danger"])

    def _apply_settings(self):
        self.detector.conf_threshold = round(self._conf_var.get(), 2)
        self.detector.imgsz = int(self._imgsz_var.get())
        self.detector.device = self._device_var.get()
        messagebox.showinfo("Settings", "Settings applied!")

    def _model_status_text(self) -> str:
        if self.detector.is_loaded():
            return f"✅ Loaded: {Path(self.detector.model_path).name}"
        return "⚠  No model — load a .onnx (Pi) or .pt file."
