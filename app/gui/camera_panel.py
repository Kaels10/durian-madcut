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
from PIL import Image, ImageTk
import cv2

from app.ml.detector import DurianDetector, CLASS_COLORS
from app.gui.theme import COLORS, FONTS
from app.utils.pi import is_raspberry_pi

# Feed display size (letterboxed)
FEED_W, FEED_H = 860, 540

# How often the UI polls for a new rendered frame (ms)
POLL_MS = 33   # ~30 fps display

# How many frames to skip between inference calls
# 0 = every frame, 1 = every other frame, etc.
INFER_EVERY_N = 2 if is_raspberry_pi() else 1


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

        self._img_tk: ImageTk.PhotoImage | None = None  # GC guard

        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self):
        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", padx=32, pady=(28, 0))

        tk.Label(header, text="📷  Live Camera Detection",
                 font=FONTS["h1"], bg=COLORS["bg"], fg=COLORS["text"]).pack(side="left")

        # Camera index picker (top-right)
        cam_sel = tk.Frame(header, bg=COLORS["bg"])
        cam_sel.pack(side="right", padx=(0, 4))
        tk.Label(cam_sel, text="Camera:", font=FONTS["small"],
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left", padx=(0, 6))
        for idx in range(4):
            tk.Radiobutton(cam_sel, text=str(idx), variable=self._cam_index,
                           value=idx, font=FONTS["small"],
                           bg=COLORS["bg"], fg=COLORS["muted"],
                           selectcolor=COLORS["card"],
                           activebackground=COLORS["bg"],
                           command=self._switch_camera).pack(side="left", padx=2)

        tk.Label(self,
                 text="Point the camera at a durian — detection runs automatically.",
                 font=FONTS["body"], bg=COLORS["bg"], fg=COLORS["muted"]).pack(
            anchor="w", padx=32, pady=(4, 12))

        # ── Body: feed (left) + sidebar (right) ─────────────────────────
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=32, pady=(0, 16))

        self._build_feed(body)
        self._build_sidebar(body)

    # ── Feed area ────────────────────────────────────────────────────────
    def _build_feed(self, parent):
        left = tk.Frame(parent, bg=COLORS["bg"])
        left.pack(side="left", fill="both", expand=True)

        # Feed canvas
        feed_frame = tk.Frame(left, bg=COLORS["card"], bd=0,
                              highlightthickness=1,
                              highlightbackground=COLORS["border"])
        feed_frame.pack(fill="both", expand=True)

        self._feed_label = tk.Label(feed_frame, bg=COLORS["card"],
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

    # ── Sidebar ──────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        right = tk.Frame(parent, bg=COLORS["bg"], width=260)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Detection Results", font=FONTS["h2"],
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", pady=(0, 8))

        self._class_widgets: dict[str, dict[str, tk.StringVar]] = {}
        for cls, cdata in CLASS_COLORS.items():
            card = tk.Frame(right, bg=COLORS["card"], bd=0,
                            highlightthickness=1,
                            highlightbackground=COLORS["border"])
            card.pack(fill="x", pady=4)

            # Colour dot
            tk.Frame(card, bg=cdata["hex"], width=10, height=10).pack(
                side="left", padx=(12, 8), pady=18)

            info = tk.Frame(card, bg=COLORS["card"])
            info.pack(side="left", fill="x", expand=True, pady=10)

            tk.Label(info, text=cls.capitalize(), font=FONTS["label"],
                     bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")

            count_var = tk.StringVar(value="0 detected")
            tk.Label(info, textvariable=count_var, font=FONTS["small"],
                     bg=COLORS["card"], fg=COLORS["muted"]).pack(anchor="w")

            self._class_widgets[cls] = {"count": count_var}

        sep = tk.Frame(right, bg=COLORS["border"], height=1)
        sep.pack(fill="x", pady=(12, 8))

        self._total_var = tk.StringVar(value="Total: 0 detections")
        tk.Label(right, textvariable=self._total_var, font=FONTS["label"],
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")

        self._model_note_var = tk.StringVar(value="⚠  No model — load one in Settings")
        self._model_note_lbl = tk.Label(right,
                                        textvariable=self._model_note_var,
                                        font=FONTS["small"],
                                        bg=COLORS["bg"], fg=COLORS["warning"],
                                        wraplength=230, justify="left")
        self._model_note_lbl.pack(anchor="w", pady=(8, 0))

        if self.detector.is_loaded():
            self._update_sidebar([])

    # ------------------------------------------------------------------
    # Camera lifecycle  (auto-start / auto-stop)
    # ------------------------------------------------------------------
    def _start_camera(self):
        idx = self._cam_index.get()
        if sys.platform == "linux":
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
                    f"No camera found at index {idx}.\n\n"
                    "• Make sure the camera is connected.\n"
                    "• Select a different camera index (0–3) above.\n"
                    "• On Raspberry Pi, enable the camera in raspi-config."
                ),
            )
            return

        if is_raspberry_pi():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
                # Still update frame so display is smooth even without new boxes
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                # Overlay last known detections
                if self._last_detections and self.detector.is_loaded():
                    pil = self.detector.draw_boxes(pil, self._last_detections)
                pil = _letterbox(pil, FEED_W, FEED_H)
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
                # Update sidebar on main thread
                self.after(0, self._update_sidebar, dets)
            else:
                self.after(0, self._model_note_var.set,
                           "⚠  No model — open Settings and load a .onnx (Pi) or .pt file")
                self.after(0, self._model_note_lbl.config, {"fg": COLORS["warning"]})

            rendered = _letterbox(pil, FEED_W, FEED_H)
            with self._lock:
                self._rendered_pil = rendered
        except Exception as exc:
            print(f"[CameraPanel] inference error: {exc}")
        finally:
            self._infer_busy = False

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
    # Sidebar updates
    # ------------------------------------------------------------------
    def _update_sidebar(self, detections: list[dict]):
        summary = self.detector.summarise(detections)
        for cls, widgets in self._class_widgets.items():
            data = summary.get(cls, {"count": 0, "avg_confidence": 0.0})
            count = data["count"]
            widgets["count"].set(f"{count} detected")
        total = len(detections)
        self._total_var.set(
            f"Total: {total} detection{'s' if total != 1 else ''}"
        )
        if self.detector.is_loaded():
            self._model_note_var.set("✅  Model active — detecting live")
            self._model_note_lbl.config(fg=COLORS["success"])

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
def _letterbox(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize preserving aspect ratio, pad with app background colour."""
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", (target_w, target_h), (15, 23, 42))
    out.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
    return out
