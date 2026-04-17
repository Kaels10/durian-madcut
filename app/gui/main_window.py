"""
app/gui/main_window.py
Root application window with sidebar navigation.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from app.ml.detector import DurianDetector
from app.gui.theme import COLORS, FONTS


class MainWindow(tk.Frame):
    def __init__(self, root: tk.Tk, detector: DurianDetector, **kwargs):
        super().__init__(root, bg=COLORS["bg"], **kwargs)
        self.root = root
        self.detector = detector
        self._active_panel = None
        self._nav_buttons: dict[str, tk.Button] = {}
        self._panels: dict[str, tk.Frame] = {}
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        self.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(self, bg=COLORS["sidebar"], width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Content area
        self._content = tk.Frame(self, bg=COLORS["bg"])
        self._content.pack(side="left", fill="both", expand=True)

        # Status bar at bottom of content
        status_bar = tk.Frame(self._content, bg=COLORS["card"], height=28)
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="⚠  No model loaded — open Settings to load a .pt file.")
        tk.Label(status_bar, textvariable=self._status_var,
                 font=FONTS["small"], bg=COLORS["card"], fg=COLORS["muted"]).pack(
            side="left", padx=12, pady=4)

        # Build panels (lazy import to avoid circular)
        self._build_panels()

        # Show default panel – camera is the landing screen
        self._show("camera")

        # Release camera when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_sidebar(self, sidebar):
        # Logo / header
        logo_frame = tk.Frame(sidebar, bg=COLORS["sidebar"])
        logo_frame.pack(fill="x", pady=0)

        tk.Label(logo_frame, text="🌾", font=("", 36),
                 bg=COLORS["sidebar"]).pack(pady=(28, 4))
        tk.Label(logo_frame, text="Durian", font=FONTS["h1"],
                 bg=COLORS["sidebar"], fg=COLORS["text"]).pack()
        tk.Label(logo_frame, text="Maturity Assessment", font=FONTS["small"],
                 bg=COLORS["sidebar"], fg=COLORS["muted"]).pack(pady=(0, 24))

        sep = tk.Frame(sidebar, bg=COLORS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=8)

        # Nav items
        nav_items = [
            ("camera",   "📷  Camera"),
            ("classify", "🔍  Classify"),
            ("batch",    "📁  Batch"),
            ("settings", "⚙  Settings"),
        ]
        for key, label in nav_items:
            btn = tk.Button(sidebar, text=label, font=FONTS["h2"],
                            anchor="w", padx=20, pady=12,
                            bg=COLORS["sidebar"], fg=COLORS["muted"],
                            activebackground=COLORS["sidebar_sel"],
                            activeforeground=COLORS["text"],
                            relief="flat", cursor="hand2",
                            command=lambda k=key: self._show(k))
            btn.pack(fill="x")
            self._nav_buttons[key] = btn

        # Bottom: quit
        tk.Frame(sidebar, bg=COLORS["sidebar"]).pack(fill="y", expand=True)
        sep2 = tk.Frame(sidebar, bg=COLORS["border"], height=1)
        sep2.pack(fill="x", padx=16, pady=8)
        tk.Button(sidebar, text="✕  Quit", font=FONTS["body"],
                  anchor="w", padx=20, pady=10,
                  bg=COLORS["sidebar"], fg=COLORS["danger"],
                  activebackground=COLORS["sidebar_sel"],
                  activeforeground=COLORS["danger"],
                  relief="flat", cursor="hand2",
                  command=self.root.quit).pack(fill="x")

    def _build_panels(self):
        from app.gui.camera_panel import CameraPanel
        from app.gui.classify_panel import ClassifyPanel
        from app.gui.batch_panel import BatchPanel
        from app.gui.settings_panel import SettingsPanel

        self._panels["camera"]   = CameraPanel(self._content, self.detector)
        self._panels["classify"] = ClassifyPanel(self._content, self.detector)
        self._panels["batch"]    = BatchPanel(self._content, self.detector)
        self._panels["settings"] = SettingsPanel(
            self._content, self.detector,
            on_model_loaded=self._on_model_loaded
        )

    # ------------------------------------------------------------------
    def _show(self, key: str):
        # Hide current
        if self._active_panel and self._active_panel in self._panels:
            self._panels[self._active_panel].pack_forget()

        # Update button styles
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.config(bg=COLORS["sidebar_sel"], fg=COLORS["text"])
            else:
                btn.config(bg=COLORS["sidebar"], fg=COLORS["muted"])

        # Show new
        self._panels[key].pack(fill="both", expand=True)
        self._active_panel = key

    def _on_model_loaded(self, path: str):
        name = Path(path).name
        self._status_var.set(f"✅ Model: {name}  ·  conf={self.detector.conf_threshold:.2f}  ·  {self.detector.device.upper()}")

    def _on_close(self):
        """Gracefully stop camera and quit."""
        try:
            cam = self._panels.get("camera")
            if cam is not None:
                cam._stop_camera()
        except Exception:
            pass
        self.root.quit()
