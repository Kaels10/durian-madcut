"""
app/gui/classify_panel.py
Single-image classification panel with bbox overlay.
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

from app.ml.detector import DurianDetector, CLASS_COLORS
from app.gui.theme import COLORS, FONTS
from app.utils.file_utils import load_pil_image, get_file_size_str

PREVIEW_MAX = (640, 480)  # max preview dimensions on-screen


class ClassifyPanel(tk.Frame):
    def __init__(self, parent, detector: DurianDetector, **kwargs):
        super().__init__(parent, bg=COLORS["bg"], **kwargs)
        self.detector = detector
        self._current_path: str = ""
        self._current_detections: list[dict] = []
        self._show_boxes = tk.BooleanVar(value=True)
        self._img_tk = None   # keep reference to prevent GC
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        # Title
        tk.Label(self, text="🔍  Classify Image", font=FONTS["h1"],
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", padx=32, pady=(32, 4))
        tk.Label(self, text="Load a single durian image to detect maturity.",
                 font=FONTS["body"], bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", padx=32)

        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=32, pady=16)

        # Left column: image
        left = tk.Frame(main, bg=COLORS["bg"])
        left.pack(side="left", fill="both", expand=True)
        self._build_image_area(left)

        # Right column: results
        right = tk.Frame(main, bg=COLORS["bg"], width=260)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)
        self._build_results_area(right)

    def _build_image_area(self, parent):
        # Toolbar
        bar = tk.Frame(parent, bg=COLORS["bg"])
        bar.pack(fill="x", pady=(0, 8))

        tk.Button(bar, text="📂  Open Image",
                  font=FONTS["h2"], bg=COLORS["accent"], fg="white",
                  activebackground=COLORS["accent_hover"], activeforeground="white",
                  relief="flat", padx=16, pady=8, cursor="hand2",
                  command=self._open_image).pack(side="left")

        tk.Checkbutton(bar, text="Show Boxes", variable=self._show_boxes,
                       font=FONTS["body"], bg=COLORS["bg"], fg=COLORS["muted"],
                       selectcolor=COLORS["card"], activebackground=COLORS["bg"],
                       command=self._refresh_preview).pack(side="left", padx=12)

        self._classify_btn = tk.Button(bar, text="▶  Classify",
                                       font=FONTS["h2"], bg=COLORS["card"], fg=COLORS["muted"],
                                       relief="flat", padx=16, pady=8,
                                       state="disabled", command=self._classify)
        self._classify_btn.pack(side="left")

        # Image canvas
        canvas_frame = tk.Frame(parent, bg=COLORS["card"], bd=0,
                                highlightthickness=1, highlightbackground=COLORS["border"])
        canvas_frame.pack(fill="both", expand=True)

        self._canvas = tk.Label(canvas_frame, bg=COLORS["card"],
                                text="Open an image to begin",
                                font=FONTS["body"], fg=COLORS["muted"])
        self._canvas.pack(fill="both", expand=True)

        # File info bar
        self._file_info_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._file_info_var,
                 font=FONTS["small"], bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", pady=4)

    def _build_results_area(self, parent):
        tk.Label(parent, text="Detection Results", font=FONTS["h2"],
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", pady=(0, 8))

        # Summary cards (one per class)
        self._class_frames = {}
        for cls, cdata in CLASS_COLORS.items():
            card = tk.Frame(parent, bg=COLORS["card"], bd=0,
                            highlightthickness=1, highlightbackground=COLORS["border"])
            card.pack(fill="x", pady=4)

            dot = tk.Frame(card, bg=cdata["hex"], width=8, height=8)
            dot.pack(side="left", padx=(12, 8), pady=16)

            info = tk.Frame(card, bg=COLORS["card"])
            info.pack(side="left", fill="x", expand=True, pady=8)

            tk.Label(info, text=cls.capitalize(), font=FONTS["label"],
                     bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")

            count_var = tk.StringVar(value="— detections")
            conf_var = tk.StringVar(value="")
            tk.Label(info, textvariable=count_var, font=FONTS["small"],
                     bg=COLORS["card"], fg=COLORS["muted"]).pack(anchor="w")
            tk.Label(info, textvariable=conf_var, font=FONTS["small"],
                     bg=COLORS["card"], fg=COLORS["muted"]).pack(anchor="w")

            self._class_frames[cls] = {"count": count_var, "conf": conf_var, "card": card}

        # Total
        sep = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep.pack(fill="x", pady=8)
        self._total_var = tk.StringVar(value="Total: — detections")
        tk.Label(parent, textvariable=self._total_var,
                 font=FONTS["label"], bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")

        # Status
        self._status_var = tk.StringVar(value="Awaiting classification…")
        self._status_label = tk.Label(parent, textvariable=self._status_var,
                                      font=FONTS["small"], bg=COLORS["bg"],
                                      fg=COLORS["muted"], wraplength=240, justify="left")
        self._status_label.pack(anchor="w", pady=8)

    # ------------------------------------------------------------------
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select durian image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self._current_path = path
        self._current_detections = []
        self._reset_results()
        self._load_preview(path, detections=None)
        info = f"{Path(path).name}  ·  {get_file_size_str(path)}"
        self._file_info_var.set(info)
        self._classify_btn.config(state="normal", bg=COLORS["accent"],
                                  fg="white", cursor="hand2")
        self._status_var.set("✅ Image loaded. Click ▶ Classify to run detection.")
        self._status_label.config(fg=COLORS["muted"])

    def _load_preview(self, path: str, detections=None):
        try:
            img = load_pil_image(path)
            if detections and self._show_boxes.get():
                from app.ml.detector import DurianDetector
                img = self.detector.draw_boxes(img, detections)
            img.thumbnail(PREVIEW_MAX, Image.LANCZOS)
            self._img_tk = ImageTk.PhotoImage(img)
            self._canvas.config(image=self._img_tk, text="")
        except Exception as e:
            self._canvas.config(image="", text=f"Error loading image:\n{e}")

    def _refresh_preview(self):
        if self._current_path:
            self._load_preview(self._current_path, self._current_detections or None)

    # ------------------------------------------------------------------
    def _classify(self):
        if not self.detector.is_loaded():
            messagebox.showwarning("No model", "Please load a YOLOv11 .pt model in Settings first.")
            return
        if not self._current_path:
            return
        self._classify_btn.config(state="disabled")
        self._status_var.set("⏳ Running detection…")
        self._status_label.config(fg=COLORS["warning"])
        self.update_idletasks()

        def _run():
            try:
                dets = self.detector.predict(self._current_path)
                self.after(0, lambda: self._on_result(dets))
            except Exception as exc:
                self.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_result(self, detections: list[dict]):
        self._current_detections = detections
        self._load_preview(self._current_path, detections)
        self._update_results(detections)
        self._classify_btn.config(state="normal")
        n = len(detections)
        self._status_var.set(f"✅ Done — {n} detection{'s' if n != 1 else ''} found.")
        self._status_label.config(fg=COLORS["success"])

    def _on_error(self, msg: str):
        self._classify_btn.config(state="normal")
        self._status_var.set(f"❌ Error: {msg}")
        self._status_label.config(fg=COLORS["danger"])

    # ------------------------------------------------------------------
    def _update_results(self, detections: list[dict]):
        summary = self.detector.summarise(detections)
        for cls, widgets in self._class_frames.items():
            data = summary.get(cls, {"count": 0, "avg_confidence": 0})
            count = data["count"]
            avg = data["avg_confidence"]
            widgets["count"].set(f"{count} detection{'s' if count != 1 else ''}")
            widgets["conf"].set(f"Avg confidence: {avg * 100:.1f}%" if count else "")
        self._total_var.set(f"Total: {len(detections)} detection{'s' if len(detections) != 1 else ''}")

    def _reset_results(self):
        for widgets in self._class_frames.values():
            widgets["count"].set("— detections")
            widgets["conf"].set("")
        self._total_var.set("Total: — detections")
