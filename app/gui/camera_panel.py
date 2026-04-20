"""
app/gui/camera_panel.py
Live camera feed with continuous real-time YOLO detection.

- Camera opens automatically when the panel is shown.
- A background thread runs YOLO on every captured frame.
- Bounding boxes + labels are overlaid directly on the live feed.
- Sidebar shows live per-class detection counts.
"""

from __future__ import annotations
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2

from app.ml.detector import DurianDetector, CLASS_COLORS
from app.gui.theme import COLORS, FONTS, UI
from app.gui.ui_components import card as ui_card, primary_button, secondary_button
from app.utils.pi import is_raspberry_pi

# How often the UI polls for a new rendered frame (ms)
POLL_MS = 33   # ~30 fps display

# How many frames to skip between inference calls
# 0 = every frame, 1 = every other frame, etc.
INFER_EVERY_N = 2 if is_raspberry_pi() else 1

# Redrawing stale boxes every preview frame is expensive and can make the
# preview feel "behind". We keep the preview as live as possible; boxes refresh
# on inference frames.
DRAW_STALE_BOXES = False

# Windows webcams often have extra buffering depending on backend + resolution.
# Using a lower capture resolution tends to reduce motion-to-glass latency.
WIN_CAP_W, WIN_CAP_H = 640, 480


class CameraPanel(tk.Frame):
    """Default landing panel — live camera + real-time detection."""

    def __init__(self, parent, detector: DurianDetector, **kwargs):
        super().__init__(parent, bg=COLORS["bg"], **kwargs)
        self.detector = detector

        self._cap: cv2.VideoCapture | None = None
        self._cam_index = tk.IntVar(value=0)

        # Shared state between capture thread and render loop
        self._lock = threading.Lock()
        self._latest_frame: cv2.typing.MatLike | None = None   # raw BGR from camera
        self._rendered_pil: Image.Image | None = None          # annotated PIL for display

        self._running = False          # camera loop alive
        self._after_id: str | None = None  # tk.after handle

        # Inference state
        self._infer_busy = False
        self._last_detections: list[dict] = []
        self._frame_count = 0

        # Sidebar totals (session counts)
        self._total_counts: dict[str, int] = {cls: 0 for cls in CLASS_COLORS.keys()}
        self._last_frame_counts: dict[str, int] = {cls: 0 for cls in CLASS_COLORS.keys()}

        self._img_tk: ImageTk.PhotoImage | None = None  # GC guard
        # Keep the camera preview at a fixed size (doesn't scale with window).
        self._feed_target_w = int(UI.get("feed_w", 860))
        self._feed_target_h = int(UI.get("feed_h", 540))
        self._feed_frame: tk.Frame | None = None
        self._body: tk.Frame | None = None

        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self):
        pad_x = int(UI["pad_x"])
        pad_top = int(UI["pad_top"])
        pad_y = int(UI["pad_y"])

        # ── Body: split layout (left: camera, right: results + controls) ──
        self._body = tk.Frame(self, bg=COLORS["bg"])
        self._body.pack(fill="both", expand=True, padx=pad_x, pady=(pad_top, pad_y))

        left = tk.Frame(self._body, bg=COLORS["bg"])
        left.pack(side="left", fill="both", expand=True)

        right_w = int(UI.get("results_w", 260))
        right = tk.Frame(self._body, bg=COLORS["bg"], width=right_w)
        right.pack(side="right", fill="y", padx=(int(UI["pad_y"]), 0))
        right.pack_propagate(False)

        self._build_feed(left, vertical=False)
        self._build_results(right)
        self._build_controls(right)

    # ── Feed area ────────────────────────────────────────────────────────
    def _build_feed(self, parent, *, vertical: bool):
        left = tk.Frame(parent, bg=COLORS["bg"])
        if vertical:
            left.pack(side="top", fill="both", expand=True)
        else:
            left.pack(side="left", fill="both", expand=True)

        # Feed canvas
        self._feed_frame = tk.Frame(
            left,
            bg=COLORS["card"],
            bd=0,
            width=self._feed_target_w,
            height=self._feed_target_h,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        # Keep the preview "as-is" even when the window is large.
        self._feed_frame.pack_propagate(False)
        self._feed_frame.pack(anchor="center", pady=(0, 0))

        self._feed_label = tk.Label(self._feed_frame, bg=COLORS["card"],
                                    fg=COLORS["muted"], font=FONTS["body"],
                                    text="Starting camera…")
        self._feed_label.pack(fill="both", expand=True)

        # Status strip beneath feed
        strip = tk.Frame(left, bg=COLORS["bg"])
        strip.pack(fill="x", pady=(4, 0))

        self._status_var = tk.StringVar(value="Initialising…")
        tk.Label(strip, textvariable=self._status_var,
                 font=FONTS["small"], bg=COLORS["bg"],
                 fg=COLORS["muted"], anchor="w").pack(side="left")

        self._fps_var = tk.StringVar(value="")
        tk.Label(strip, textvariable=self._fps_var,
                 font=FONTS["small"], bg=COLORS["bg"],
                 fg=COLORS["muted"], anchor="e").pack(side="right")
        return left

    # ── Results column ───────────────────────────────────────────────────
    def _build_results(self, parent: tk.Widget) -> None:
        body = ui_card(parent, "Maturity Result")

        self._result_label_var = tk.StringVar(value="—")
        self._result_detail_var = tk.StringVar(value="Waiting for detections…")

        self._result_badge = tk.Label(
            body,
            textvariable=self._result_label_var,
            font=FONTS["result"],
            bg=COLORS["card"],
            fg=COLORS["text"],
            padx=12,
            pady=14,
        )
        self._result_badge.pack(fill="x")

        tk.Label(
            body,
            textvariable=self._result_detail_var,
            font=FONTS["body"],
            bg=COLORS["card"],
            fg=COLORS["muted"],
            justify="left",
            wraplength=int(UI.get("results_w", 260)) - 40,
        ).pack(anchor="w", pady=(10, 0))

        # Live counts
        counts = tk.Frame(body, bg=COLORS["card"])
        counts.pack(fill="x", pady=(12, 0))
        self._count_vars = {
            "mature": tk.StringVar(value="0"),
            "immature": tk.StringVar(value="0"),
            "damaged": tk.StringVar(value="0"),
        }
        for key, label in (("mature", "Mature"), ("immature", "Immature"), ("damaged", "Damaged")):
            row = tk.Frame(counts, bg=COLORS["card"])
            row.pack(fill="x", pady=3)
            dot = tk.Label(row, text="●", font=FONTS["body"], bg=COLORS["card"], fg=COLORS[key])
            dot.pack(side="left")
            tk.Label(row, text=f" {label}", font=FONTS["label"], bg=COLORS["card"], fg=COLORS["text"]).pack(
                side="left"
            )
            tk.Label(row, textvariable=self._count_vars[key], font=FONTS["label"], bg=COLORS["card"], fg=COLORS[key]).pack(
                side="right"
            )

    def _build_controls(self, parent: tk.Widget) -> None:
        body = ui_card(parent, "Controls")

        # Camera selector (compact + touch-friendly)
        tk.Label(
            body,
            text="Camera",
            font=FONTS["label"],
            bg=COLORS["card"],
            fg=COLORS["muted"],
        ).pack(anchor="w")

        cam_row = tk.Frame(body, bg=COLORS["card"])
        cam_row.pack(fill="x", pady=(8, 10))

        cam_opts = [0, 1, 2, 3]
        cam_menu = tk.OptionMenu(cam_row, self._cam_index, *cam_opts)
        cam_menu.config(
            font=FONTS["body"],
            bg=COLORS["input"],
            fg=COLORS["text"],
            activebackground=COLORS["card_hover"],
            activeforeground=COLORS["text"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            relief="flat",
            padx=10,
            pady=max(6, int(UI.get("input_pady", 10)) - 2),
        )
        cam_menu["menu"].config(
            bg=COLORS["card"],
            fg=COLORS["text"],
            activebackground=COLORS["sidebar_sel"],
            activeforeground=COLORS["text"],
            relief="flat",
        )
        cam_menu.pack(side="left", fill="x", expand=True)

        self._switch_btn = primary_button(body, "Switch Camera", command=self._switch_camera)
        self._switch_btn.pack(fill="x", pady=(0, 10))

        # Pause / Resume
        self._pause_btn = secondary_button(body, "Pause", command=self._toggle_pause)
        self._pause_btn.pack(fill="x")

    def _toggle_pause(self) -> None:
        if self._running:
            self._stop_camera()
            self._status_var.set("⏸  Paused")
            self._pause_btn.config(text="Resume")
            return
        self._pause_btn.config(text="Pause")
        self._start_camera()

    # ------------------------------------------------------------------
    # Camera lifecycle  (auto-start / auto-stop)
    # ------------------------------------------------------------------
    def _start_camera(self):
        idx = self._cam_index.get()
        # Prefer low-latency backend per platform.
        if sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        elif sys.platform == "linux":
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not cap.isOpened():
            self._status_var.set(
                f"⚠  Could not open camera {idx}. Try a different index."
            )
            self._feed_label.config(
                image="",
                text=(
                    "No camera feed\n\n"
                    f"Selected: camera {idx}\n\n"
                    "Try:\n"
                    "• Plug in / enable the camera\n"
                    "• Pick another camera index\n"
                    "• On Raspberry Pi: enable camera in raspi-config"
                ),
            )
            return

        # Reduce capture latency where supported (some backends ignore these).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        # Request MJPG where supported (can reduce latency on some webcams).
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        if is_raspberry_pi():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        elif sys.platform == "win32":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_CAP_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_CAP_H)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._cap = cap
        self._running = True
        self._status_var.set(f"📡  Camera {idx} active — detection running…")
        self._fps_var.set("")

        # Launch capture thread
        threading.Thread(target=self._capture_loop, daemon=True).start()
        # Start UI render loop
        self._render_loop()

    def _stop_camera(self):
        self._running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest_frame = None
            self._rendered_pil = None

    def _switch_camera(self):
        self._stop_camera()
        self._start_camera()

    # ------------------------------------------------------------------
    # Capture thread — reads frames from the camera as fast as possible
    # ------------------------------------------------------------------
    def _capture_loop(self):
        t_last = time.perf_counter()
        frame_n = 0

        while self._running and self._cap is not None:
            # Grab/retrieve pattern helps drop buffered frames (lower latency).
            # If grab() isn't supported by the backend, read() still works.
            frame = None
            ret = False
            try:
                # Drop a couple buffered frames when possible.
                for _ in range(2):
                    if not self._cap.grab():
                        break
                ret, frame = self._cap.retrieve()
            except Exception:
                ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_n += 1

            # Dispatch inference every INFER_EVERY_N frames
            if frame_n % (INFER_EVERY_N + 1) == 0 and not self._infer_busy:
                self._infer_busy = True
                threading.Thread(
                    target=self._infer_and_render,
                    args=(frame.copy(),),
                    daemon=True,
                ).start()
            else:
                # Keep preview as "live" as possible on non-inference frames.
                pil = _preview_from_bgr(frame, self._feed_target_w, self._feed_target_h)
                if DRAW_STALE_BOXES and self._last_detections and self.detector.is_loaded():
                    pil = self.detector.draw_boxes(pil, self._last_detections)
                with self._lock:
                    self._rendered_pil = pil

            # FPS
            now = time.perf_counter()
            elapsed = now - t_last
            t_last = now
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.after(0, self._fps_var.set, f"Camera: {fps:.1f} fps")

    # ------------------------------------------------------------------
    # Inference + render (runs in its own thread per invocation)
    # ------------------------------------------------------------------
    def _infer_and_render(self, bgr_frame):
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            if self.detector.is_loaded():
                dets = self.detector.predict_pil(pil)
                self._last_detections = dets
                if dets:
                    pil = self.detector.draw_boxes(pil, dets)
                self.after(0, self._update_results_ui, dets)
            else:
                self.after(0, self._update_results_ui, [])

            rendered = _preview_from_pil(pil, self._feed_target_w, self._feed_target_h)
            with self._lock:
                self._rendered_pil = rendered
        except Exception as exc:
            print(f"[CameraPanel] inference error: {exc}")
        finally:
            self._infer_busy = False

    def _update_results_ui(self, detections: list[dict]) -> None:
        # Count classes
        counts = {"mature": 0, "immature": 0, "damaged": 0}
        for d in detections or []:
            lbl = str(d.get("label", "")).strip().lower()
            if lbl in counts:
                counts[lbl] += 1

        for k, v in counts.items():
            if hasattr(self, "_count_vars"):
                self._count_vars[k].set(str(v))

        total = sum(counts.values())
        if total == 0:
            self._set_result_state("—", COLORS["card"], COLORS["text"], "No detections yet.")
            return

        # Decide primary result by majority; tie-breaker: damaged > immature > mature
        order = ["damaged", "immature", "mature"]
        best = max(order, key=lambda k: (counts[k], -order.index(k)))

        if best == "mature":
            self._set_result_state("Matured", COLORS["mature"], "white", f"Detected {total} object(s).")
        elif best == "immature":
            self._set_result_state("Immature", COLORS["immature"], "#111111", f"Detected {total} object(s).")
        else:
            self._set_result_state("Damaged", COLORS["damaged"], "white", f"Detected {total} object(s).")

    def _set_result_state(self, title: str, bg: str, fg: str, detail: str) -> None:
        if not hasattr(self, "_result_badge"):
            return
        self._result_label_var.set(title)
        self._result_detail_var.set(detail)
        self._result_badge.config(bg=bg, fg=fg)

    # ------------------------------------------------------------------
    # UI render loop (tk.after, main thread)
    # ------------------------------------------------------------------
    def _render_loop(self):
        if not self._running:
            return

        with self._lock:
            frame_pil = self._rendered_pil

        if frame_pil is not None:
            self._img_tk = ImageTk.PhotoImage(frame_pil)
            self._feed_label.config(image=self._img_tk, text="")

        self._after_id = self.after(POLL_MS, self._render_loop)

    # ------------------------------------------------------------------
    # Panel show / hide hooks  (called by MainWindow._show)
    # ------------------------------------------------------------------
    def pack(self, **kwargs):
        """Auto-start camera when the panel becomes visible."""
        super().pack(**kwargs)
        if not self._running:
            self._start_camera()

    def pack_forget(self):
        """Auto-stop camera when navigating away."""
        self._stop_camera()
        super().pack_forget()

    # Graceful cleanup on window close
    def destroy(self):
        self._stop_camera()
        super().destroy()


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def _bg_rgb() -> tuple[int, int, int]:
    bg = COLORS.get("bg", "#000000").lstrip("#")
    try:
        return tuple(int(bg[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return (0, 0, 0)


def _letterbox_bgr(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Fast letterbox in OpenCV space (BGR).
    Returns a target_w x target_h BGR frame.
    """
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return frame_bgr
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left

    r, g, b = _bg_rgb()
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(b, g, r),  # OpenCV uses BGR
    )


def _preview_from_bgr(frame_bgr: np.ndarray, target_w: int, target_h: int) -> Image.Image:
    """
    Convert a live camera BGR frame into a display-ready PIL image.
    Uses OpenCV for resize/pad to keep CPU low and latency down.
    """
    boxed = _letterbox_bgr(frame_bgr, target_w, target_h)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _preview_from_pil(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Letterbox a PIL image for display.
    (Used for inference frames after drawing boxes.)
    """
    rgb_np = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    boxed = _letterbox_bgr(bgr, target_w, target_h)
    rgb2 = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb2)
