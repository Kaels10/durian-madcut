"""
app/gui/main_window.py
Root application window with sidebar navigation.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from app.ml.detector import DurianDetector
from app.gui.theme import COLORS, FONTS, UI


class MainWindow(tk.Frame):
    def __init__(self, root: tk.Tk, detector: DurianDetector, **kwargs):
        super().__init__(root, bg=COLORS["bg"], **kwargs)
        self.root = root
        self.detector = detector
        self._active_panel = None
        self._nav_buttons: dict[str, tk.Button] = {}
        self._panels: dict[str, tk.Frame] = {}
        self._sidebar_visible = True
        self._sidebar: tk.Frame | None = None
        self._sidebar_toggle_btn: tk.Button | None = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        self.pack(fill="both", expand=True)

        # Left: always-visible handle + collapsible sidebar
        left = tk.Frame(self, bg=COLORS["sidebar"])
        left.pack(side="left", fill="y")

        handle_w = 28 if UI.get("is_small_screen") else 32
        handle = tk.Frame(left, bg=COLORS["sidebar"], width=handle_w)
        handle.pack(side="left", fill="y")
        handle.pack_propagate(False)

        self._sidebar_toggle_btn = tk.Button(
            handle,
            text="◀",
            font=FONTS["h2"],
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            activebackground=COLORS["sidebar_sel"],
            activeforeground=COLORS["text"],
            relief="flat",
            cursor="hand2",
            command=self._toggle_sidebar,
        )
        self._sidebar_toggle_btn.pack(side="top", fill="x", pady=(10, 0))

        self._sidebar = tk.Frame(left, bg=COLORS["sidebar"], width=int(UI["sidebar_w"]))
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)
        self._build_sidebar(self._sidebar)

        # Content area
        self._content = tk.Frame(self, bg=COLORS["bg"])
        self._content.pack(side="left", fill="both", expand=True)

        # Header bar (global)
        header = tk.Frame(self._content, bg=COLORS["card"])
        header.pack(side="top", fill="x")
        title = tk.Frame(header, bg=COLORS["card"])
        title.pack(side="left", padx=16, pady=12)
        tk.Label(
            title,
            text="MAD-CUT",
            font=FONTS["h1"],
            bg=COLORS["card"],
            fg=COLORS["text"],
        ).pack(anchor="w")
        tk.Label(
            title,
            text="Maturity Assessment Durian Cutter",
            font=FONTS["body"],
            bg=COLORS["card"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(2, 0))

        # Main panel container (between header + status bar)
        self._main = tk.Frame(self._content, bg=COLORS["bg"])
        self._main.pack(side="top", fill="both", expand=True)

        # Status bar at bottom of content
        status_bar = tk.Frame(self._content, bg=COLORS["card"], height=28)
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)
        self._status_var = tk.StringVar(value=self._initial_status_text())
        tk.Label(status_bar, textvariable=self._status_var,
                 font=FONTS["small"], bg=COLORS["card"], fg=COLORS["muted"]).pack(
            side="left", padx=12, pady=4)

        # Build panels (lazy import to avoid circular)
        self._build_panels()

        # Show default panel – camera is the landing screen
        self._show("camera")

        # Release camera when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_sidebar(self):
        if self._sidebar is None or self._sidebar_toggle_btn is None:
            return
        if self._sidebar_visible:
            self._sidebar.pack_forget()
            self._sidebar_toggle_btn.config(text="▶")
            self._sidebar_visible = False
        else:
            self._sidebar.pack(side="left", fill="y")
            self._sidebar_toggle_btn.config(text="◀")
            self._sidebar_visible = True

    def _initial_status_text(self) -> str:
        if self.detector.is_loaded() and self.detector.model_path:
            name = Path(self.detector.model_path).name
            return (
                f"✅ Model: {name}  ·  {self.detector.runtime_label}  ·  "
                f"conf={self.detector.conf_threshold:.2f}"
            )
        return (
            "⚠  No model — add models/best.onnx or set DURIAN_MODEL, "
            "or open Settings to browse for a .onnx / .pt file."
        )

    def _build_sidebar(self, sidebar):
        # Keep the sidebar header area clean (no logo/branding).
        tk.Frame(sidebar, bg=COLORS["sidebar"], height=24).pack(fill="x")

        sep = tk.Frame(sidebar, bg=COLORS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=8)

        # Nav items
        nav_items = [
            ("camera",   "📷  Camera"),
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
        from app.gui.settings_panel import SettingsPanel

        self._panels["camera"]   = CameraPanel(self._main, self.detector)
        self._panels["settings"] = SettingsPanel(
            self._main, self.detector,
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
        self._status_var.set(
            f"✅ Model: {name}  ·  {self.detector.runtime_label}  ·  conf={self.detector.conf_threshold:.2f}"
        )

    def _on_close(self):
        """Gracefully stop camera and quit."""
        try:
            cam = self._panels.get("camera")
            if cam is not None:
                cam._stop_camera()
        except Exception:
            pass
        self.root.quit()
