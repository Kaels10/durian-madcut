"""
app/gui/main_window.py
Root application window with sidebar navigation.
"""

from __future__ import annotations
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from app.ml.detector import DurianDetector
from app.gui.theme import COLORS, FONTS, UI
from app.utils.pi import is_raspberry_pi


class MainWindow(tk.Frame):
    def __init__(self, root: tk.Tk, detector: DurianDetector, **kwargs):
        super().__init__(root, bg=COLORS["bg"], **kwargs)
        self.root = root
        self.detector = detector
        self._active_panel = None
        self._nav_buttons: dict[str, tk.Button] = {}
        self._nav_indicators: dict[str, tk.Frame] = {}
        self._panels: dict[str, tk.Frame] = {}
        self._sidebar_visible = True
        self._sidebar: tk.Frame | None = None
        self._sidebar_toggle_btn: tk.Button | None = None
        self._sidebar_logo_photo: ImageTk.PhotoImage | None = None
        self._pi_app_chrome = is_raspberry_pi()
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

        if self._pi_app_chrome:
            self._build_pi_toolbar(self._content)
        else:
            header = tk.Frame(self._content, bg=COLORS["card"])
            header.pack(side="top", fill="x")
            title = tk.Frame(header, bg=COLORS["card"])
            title.pack(side="left", padx=16, pady=(12, 10))
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

        # Main panel container (between header/toolbar + status bar)
        self._main = tk.Frame(self._content, bg=COLORS["bg"])
        self._main.pack(side="top", fill="both", expand=True)

        # Status bar at bottom of content
        status_bar = tk.Frame(self._content, bg=COLORS["card"], height=28)
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)
        self._status_var = tk.StringVar(value=self._initial_status_text())
        self._status_right_var = tk.StringVar(value="")
        tk.Label(
            status_bar,
            textvariable=self._status_var,
            font=FONTS["small"],
            bg=COLORS["card"],
            fg=COLORS["muted"],
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=12, pady=4)
        tk.Label(
            status_bar,
            textvariable=self._status_right_var,
            font=FONTS["small"],
            bg=COLORS["card"],
            fg=COLORS["muted"],
            anchor="e",
        ).pack(side="right", padx=12, pady=4)

        # Build panels (lazy import to avoid circular)
        self._build_panels()

        # Show default panel – camera is the landing screen
        self._show("camera")

        # Release camera when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_pi_toolbar(self, parent: tk.Frame) -> None:
        """Raspberry Pi: Camera / Settings / Quit above main content."""
        bar = tk.Frame(parent, bg=COLORS["card"])
        bar.pack(side="top", fill="x")
        inner = tk.Frame(bar, bg=COLORS["card"])
        inner.pack(fill="x", padx=12, pady=10)

        nav_host = tk.Frame(inner, bg=COLORS["card"])
        nav_host.pack(side="left")

        indicator_w = 4
        nav_items = [
            ("camera", "📷  Camera"),
            ("settings", "⚙  Settings"),
        ]
        nav_pady = 8 if UI.get("is_small_screen") else 6
        nav_padx = 14 if UI.get("is_small_screen") else 16

        for key, label in nav_items:
            row = tk.Frame(nav_host, bg=COLORS["card"])
            row.pack(side="left", padx=(0, 10))
            ind = tk.Frame(row, bg=COLORS["card"], width=indicator_w)
            ind.pack(side="left", fill="y")
            ind.pack_propagate(False)
            self._nav_indicators[key] = ind
            btn = tk.Button(
                row,
                text=label,
                font=FONTS["h2"],
                anchor="w",
                padx=nav_padx,
                pady=nav_pady,
                bg=COLORS["card"],
                fg=COLORS["muted"],
                activebackground=COLORS["card_hover"],
                activeforeground=COLORS["text"],
                relief="flat",
                cursor="hand2",
                command=lambda k=key: self._show(k),
            )
            btn.pack(side="left")
            self._nav_buttons[key] = btn

        tk.Frame(inner, bg=COLORS["card"]).pack(side="left", fill="x", expand=True)

        tk.Button(
            inner,
            text="✕  Quit",
            font=FONTS["body"],
            anchor="e",
            padx=16,
            pady=nav_pady,
            bg=COLORS["card"],
            fg=COLORS["danger"],
            activebackground=COLORS["card_hover"],
            activeforeground=COLORS["danger"],
            relief="flat",
            cursor="hand2",
            command=self.root.quit,
        ).pack(side="right")

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
        indicator_w = 4

        logo_path = Path(__file__).resolve().parent.parent.parent / "assets" / "madcut_logo.png"

        if self._pi_app_chrome:
            self._build_pi_sidebar_polished(sidebar, logo_path, indicator_w)
            return

        logo_row = tk.Frame(sidebar, bg=COLORS["sidebar"])
        logo_row.pack(fill="x")

        logo_gutter = tk.Frame(logo_row, bg=COLORS["sidebar"], width=indicator_w)
        logo_gutter.pack(side="left", fill="y")
        logo_gutter.pack_propagate(False)

        logo_cell = tk.Frame(logo_row, bg=COLORS["sidebar"])
        logo_cell.pack(side="left", fill="x", expand=True)

        self._pack_sidebar_logo(logo_cell, logo_path, indicator_w)

        sep = tk.Frame(sidebar, bg=COLORS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=(4, 8))

        nav_items = [
            ("camera", "📷  Camera"),
            ("settings", "⚙  Settings"),
        ]

        nav_pady = 12 if UI.get("is_small_screen") else 10
        nav_padx = 18 if UI.get("is_small_screen") else 20

        for key, label in nav_items:
            row = tk.Frame(sidebar, bg=COLORS["sidebar"])
            row.pack(fill="x")

            ind = tk.Frame(row, bg=COLORS["sidebar"], width=indicator_w)
            ind.pack(side="left", fill="y")
            ind.pack_propagate(False)
            self._nav_indicators[key] = ind

            btn = tk.Button(
                row,
                text=label,
                font=FONTS["h2"],
                anchor="w",
                padx=nav_padx,
                pady=nav_pady,
                bg=COLORS["sidebar"],
                fg=COLORS["muted"],
                activebackground=COLORS["sidebar_sel"],
                activeforeground=COLORS["text"],
                relief="flat",
                cursor="hand2",
                command=lambda k=key: self._show(k),
            )
            btn.pack(side="left", fill="x", expand=True)
            self._nav_buttons[key] = btn

        tk.Frame(sidebar, bg=COLORS["sidebar"]).pack(fill="y", expand=True)
        sep2 = tk.Frame(sidebar, bg=COLORS["border"], height=1)
        sep2.pack(fill="x", padx=16, pady=8)
        tk.Button(
            sidebar,
            text="✕  Quit",
            font=FONTS["body"],
            anchor="w",
            padx=20,
            pady=10,
            bg=COLORS["sidebar"],
            fg=COLORS["danger"],
            activebackground=COLORS["sidebar_sel"],
            activeforeground=COLORS["danger"],
            relief="flat",
            cursor="hand2",
            command=self.root.quit,
        ).pack(fill="x")

    def _pack_sidebar_logo(
        self,
        logo_cell: tk.Frame,
        logo_path: Path,
        indicator_w: int,
        *,
        content_bg: str | None = None,
        max_width_slack: int = 0,
    ) -> None:
        """Load and show MAD-CUT logo in logo_cell (non-Pi: sidebar bg; Pi: plate bg)."""
        bg = content_bg or COLORS["sidebar"]
        if logo_path.is_file():
            try:
                src = Image.open(logo_path).convert("RGBA")
                ow, oh = src.size
                if ow < 1 or oh < 1:
                    raise ValueError("invalid logo dimensions")
                inner_pad = 10
                max_w = max(1, int(UI["sidebar_w"]) - indicator_w - inner_pad - max_width_slack)
                max_h = 96 if UI.get("is_small_screen") else 132
                scale = min(max_w / ow, max_h / oh, 1.0)
                w = max(1, int(round(ow * scale)))
                h = max(1, int(round(oh * scale)))

                img = src.resize((w, h), Image.Resampling.LANCZOS)
                self._sidebar_logo_photo = ImageTk.PhotoImage(img)
                tk.Label(
                    logo_cell,
                    image=self._sidebar_logo_photo,
                    bg=bg,
                ).pack(pady=(10, 6))
            except (OSError, ValueError, tk.TclError):
                self._sidebar_logo_photo = None

    def _build_pi_sidebar_polished(
        self,
        sidebar: tk.Frame,
        logo_path: Path,
        indicator_w: int,
    ) -> None:
        """
        Raspberry Pi: branded rail — left accent strip, inset logo/title plate,
        right vertical rail against the main content area.
        """
        rail_w = 3
        strip_w = 3
        rail_bg = COLORS.get("sidebar_rail", COLORS["accent"])

        right_rail = tk.Frame(sidebar, bg=rail_bg, width=rail_w)
        right_rail.pack(side="right", fill="y")
        right_rail.pack_propagate(False)

        work = tk.Frame(sidebar, bg=COLORS["sidebar"])
        work.pack(side="left", fill="both", expand=True)

        left_strip = tk.Frame(work, bg=COLORS["accent"], width=strip_w)
        left_strip.pack(side="left", fill="y")
        left_strip.pack_propagate(False)

        body = tk.Frame(work, bg=COLORS["sidebar"])
        body.pack(side="left", fill="both", expand=True)

        logo_row = tk.Frame(body, bg=COLORS["sidebar"])
        logo_row.pack(fill="x")

        plate_bg = COLORS["sidebar_sel"]
        logo_cell = tk.Frame(
            logo_row,
            bg=plate_bg,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        logo_cell.pack(fill="x", padx=(6, 8), pady=(10, 8))

        # Narrower effective width: strips + plate padding + border
        slack = rail_w + strip_w + 28
        self._pack_sidebar_logo(
            logo_cell,
            logo_path,
            indicator_w,
            content_bg=plate_bg,
            max_width_slack=slack,
        )

        title_wrap = max(60, int(UI["sidebar_w"]) - indicator_w - slack)
        tk.Label(
            logo_cell,
            text="MAD-CUT",
            font=FONTS["h2"],
            bg=plate_bg,
            fg=COLORS["text"],
            wraplength=title_wrap,
            justify="left",
        ).pack(anchor="w", padx=(8, 8))
        tk.Label(
            logo_cell,
            text="Maturity Assessment Durian Cutter",
            font=FONTS["small"],
            bg=plate_bg,
            fg=COLORS["muted"],
            wraplength=title_wrap,
            justify="left",
        ).pack(anchor="w", padx=(8, 8), pady=(2, 4))

        rule = tk.Frame(body, bg=COLORS["border"], height=1)
        rule.pack(fill="x", padx=(10, 10), pady=(4, 0))

        tk.Frame(body, bg=COLORS["sidebar"]).pack(fill="y", expand=True)

    def _build_panels(self):
        from app.gui.camera_panel import CameraPanel
        from app.gui.settings_panel import SettingsPanel

        self._panels["camera"] = CameraPanel(self._main, self.detector)
        self._panels["settings"] = SettingsPanel(
            self._main,
            self.detector,
            on_model_loaded=self._on_model_loaded,
        )

    # ------------------------------------------------------------------
    def _show(self, key: str):
        if self._active_panel and self._active_panel in self._panels:
            self._panels[self._active_panel].pack_forget()

        for k, btn in self._nav_buttons.items():
            if self._pi_app_chrome:
                if k == key:
                    btn.config(bg=COLORS["card_hover"], fg=COLORS["text"])
                    if k in self._nav_indicators:
                        self._nav_indicators[k].config(bg=COLORS["accent"])
                else:
                    btn.config(bg=COLORS["card"], fg=COLORS["muted"])
                    if k in self._nav_indicators:
                        self._nav_indicators[k].config(bg=COLORS["card"])
            else:
                if k == key:
                    btn.config(bg=COLORS["sidebar"], fg=COLORS["text"])
                    if k in self._nav_indicators:
                        self._nav_indicators[k].config(bg=COLORS["accent"])
                else:
                    btn.config(bg=COLORS["sidebar"], fg=COLORS["muted"])
                    if k in self._nav_indicators:
                        self._nav_indicators[k].config(bg=COLORS["sidebar"])

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
