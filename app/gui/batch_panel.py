"""
app/gui/batch_panel.py
Batch processing panel: run inference on all images in a folder.
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk

from app.ml.detector import DurianDetector, CLASS_COLORS
from app.gui.theme import COLORS, FONTS
from app.utils.file_utils import list_images, load_pil_image
from app.utils.export import export_csv, generate_csv_filename

THUMB_SIZE = (80, 60)


class BatchPanel(tk.Frame):
    def __init__(self, parent, detector: DurianDetector, **kwargs):
        super().__init__(parent, bg=COLORS["bg"], **kwargs)
        self.detector = detector
        self._folder: str = ""
        self._results: list[dict] = []
        self._running = False
        self._thumbs: list[ImageTk.PhotoImage] = []  # GC guard
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        tk.Label(self, text="📁  Batch Process", font=FONTS["h1"],
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", padx=32, pady=(32, 4))
        tk.Label(self, text="Select a folder of images and run detection on all of them.",
                 font=FONTS["body"], bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", padx=32)

        # Toolbar
        bar = tk.Frame(self, bg=COLORS["bg"])
        bar.pack(fill="x", padx=32, pady=12)

        tk.Button(bar, text="📂  Select Folder",
                  font=FONTS["h2"], bg=COLORS["accent"], fg="white",
                  activebackground=COLORS["accent_hover"], activeforeground="white",
                  relief="flat", padx=16, pady=8, cursor="hand2",
                  command=self._browse).pack(side="left")

        self._run_btn = tk.Button(bar, text="▶  Run Batch",
                                  font=FONTS["h2"], bg=COLORS["card"], fg=COLORS["muted"],
                                  relief="flat", padx=16, pady=8, state="disabled",
                                  command=self._run_batch)
        self._run_btn.pack(side="left", padx=8)

        self._export_btn = tk.Button(bar, text="💾  Export CSV",
                                     font=FONTS["h2"], bg=COLORS["card"], fg=COLORS["muted"],
                                     relief="flat", padx=16, pady=8, state="disabled",
                                     command=self._export_csv)
        self._export_btn.pack(side="left", padx=4)

        self._folder_var = tk.StringVar(value="No folder selected")
        tk.Label(bar, textvariable=self._folder_var,
                 font=FONTS["small"], bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left", padx=16)

        # Progress bar
        prog_frame = tk.Frame(self, bg=COLORS["bg"])
        prog_frame.pack(fill="x", padx=32, pady=(0, 8))
        self._progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                         mode="determinate", length=400)
        self._progress.pack(side="left", fill="x", expand=True)
        self._progress_var = tk.StringVar(value="")
        tk.Label(prog_frame, textvariable=self._progress_var,
                 font=FONTS["small"], bg=COLORS["bg"], fg=COLORS["muted"],
                 width=20).pack(side="left", padx=8)

        # Summary bar
        self._summary_frame = tk.Frame(self, bg=COLORS["bg"])
        self._summary_frame.pack(fill="x", padx=32, pady=(0, 8))
        self._summary_labels: dict[str, tk.Label] = {}
        for cls, cdata in CLASS_COLORS.items():
            lbl = tk.Label(self._summary_frame, text=f"{cls.capitalize()}: 0",
                           font=FONTS["label"], bg=COLORS["bg"], fg=cdata["hex"])
            lbl.pack(side="left", padx=12)
            self._summary_labels[cls] = lbl

        # Results table
        table_frame = tk.Frame(self, bg=COLORS["card"])
        table_frame.pack(fill="both", expand=True, padx=32, pady=(0, 16))
        self._build_table(table_frame)

    def _build_table(self, parent):
        cols = ("image", "label", "confidence", "detections", "timestamp")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Batch.Treeview",
                        background=COLORS["card"],
                        foreground=COLORS["text"],
                        rowheight=28,
                        fieldbackground=COLORS["card"],
                        bordercolor=COLORS["border"],
                        borderwidth=0,
                        font=FONTS["body"])
        style.configure("Batch.Treeview.Heading",
                        background=COLORS["bg"],
                        foreground=COLORS["muted"],
                        relief="flat",
                        font=FONTS["label"])
        style.map("Batch.Treeview",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "white")])

        self._tree = ttk.Treeview(parent, columns=cols, show="headings",
                                  style="Batch.Treeview", selectmode="browse")
        col_widths = {"image": 260, "label": 100, "confidence": 120, "detections": 100, "timestamp": 160}
        for col in cols:
            self._tree.heading(col, text=col.capitalize())
            self._tree.column(col, width=col_widths[col], anchor="center" if col != "image" else "w")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Tag colours per class
        self._tree.tag_configure("mature",   foreground=COLORS["mature"])
        self._tree.tag_configure("immature", foreground=COLORS["immature"])
        self._tree.tag_configure("damaged",  foreground=COLORS["damaged"])
        self._tree.tag_configure("none",     foreground=COLORS["muted"])

    # ------------------------------------------------------------------
    def _browse(self):
        folder = filedialog.askdirectory(title="Select folder with durian images")
        if not folder:
            return
        self._folder = folder
        imgs = list_images(folder)
        self._folder_var.set(f"{Path(folder).name}  ·  {len(imgs)} image(s)")
        self._run_btn.config(state="normal", bg=COLORS["accent"], fg="white", cursor="hand2")

    def _run_batch(self):
        if not self.detector.is_loaded():
            messagebox.showwarning("No model", "Please load a YOLOv11 .pt model in Settings first.")
            return
        if self._running:
            return
        imgs = list_images(self._folder)
        if not imgs:
            messagebox.showinfo("No images", "No images found in the selected folder.")
            return

        self._results = []
        self._tree.delete(*self._tree.get_children())
        self._running = True
        self._run_btn.config(state="disabled")
        self._export_btn.config(state="disabled", bg=COLORS["card"], fg=COLORS["muted"])
        self._progress["maximum"] = len(imgs)
        self._progress["value"] = 0

        def _worker():
            for i, img_path in enumerate(imgs, start=1):
                try:
                    dets = self.detector.predict(img_path)
                except Exception as exc:
                    dets = []
                    print(f"[Batch] Error on {img_path}: {exc}")

                ts = datetime.now().strftime("%H:%M:%S")
                # Aggregate top label (most frequent / highest conf)
                if dets:
                    top = max(dets, key=lambda d: d["confidence"])
                    top_label = top["label"]
                    top_conf = top["confidence"]
                else:
                    top_label = "none"
                    top_conf = 0.0

                row = {
                    "filename":    img_path,
                    "label":       top_label,
                    "confidence":  top_conf,
                    "detections":  len(dets),
                    "timestamp":   ts,
                    "all_dets":    dets,
                }
                self._results.append(row)
                self.after(0, self._add_row, row, i, len(imgs))
            self.after(0, self._on_batch_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _add_row(self, row: dict, progress: int, total: int):
        self._progress["value"] = progress
        self._progress_var.set(f"{progress}/{total}")
        tag = row["label"].lower() if row["label"].lower() in CLASS_COLORS else "none"
        conf_str = f"{row['confidence'] * 100:.1f}%" if row["detections"] else "—"
        self._tree.insert("", "end", values=(
            Path(row["filename"]).name,
            row["label"].capitalize() if row["label"] != "none" else "No detection",
            conf_str,
            row["detections"],
            row["timestamp"],
        ), tags=(tag,))
        self._tree.yview_moveto(1.0)
        self._update_summary()

    def _on_batch_done(self):
        self._running = False
        self._run_btn.config(state="normal")
        n = len(self._results)
        self._progress_var.set(f"Done — {n} images")
        if self._results:
            self._export_btn.config(state="normal", bg=COLORS["warning"], fg="white", cursor="hand2")

    def _update_summary(self):
        counts = {cls: 0 for cls in CLASS_COLORS}
        for r in self._results:
            lbl = r["label"].lower()
            if lbl in counts:
                counts[lbl] += 1
        for cls, lbl in self._summary_labels.items():
            lbl.config(text=f"{cls.capitalize()}: {counts[cls]}")

    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self._results:
            return
        # Flatten: one row per detection; if no detections, one row per image
        flat = []
        for r in self._results:
            if r["all_dets"]:
                for det in r["all_dets"]:
                    flat.append({
                        "filename":   r["filename"],
                        "label":      det["label"],
                        "confidence": det["confidence"],
                        "bbox":       det["bbox"],
                        "timestamp":  r["timestamp"],
                    })
            else:
                flat.append({
                    "filename":   r["filename"],
                    "label":      "none",
                    "confidence": 0.0,
                    "bbox":       (0, 0, 0, 0),
                    "timestamp":  r["timestamp"],
                })

        default_name = generate_csv_filename(self._folder or ".")
        filepath = filedialog.asksaveasfilename(
            title="Save results as CSV",
            defaultextension=".csv",
            initialfile=Path(default_name).name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not filepath:
            return
        try:
            count = export_csv(flat, filepath)
            messagebox.showinfo("Export", f"Saved {count} rows to:\n{filepath}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
