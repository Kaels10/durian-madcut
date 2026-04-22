"""
app/gui/camera_panel.py
Live camera feed with continuous real-time YOLO detection.

- Camera opens automatically when the panel is shown.
- On Raspberry Pi, the camera preview updates every frame while YOLO runs
  asynchronously (same model imgsz); other platforms use the legacy fused path.
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
POLL_MS = 16 if is_raspberry_pi() else 33  # Pi: faster poll when preview is decoupled from ONNX

# Raspberry Pi: camera letterbox runs every frame; ONNX runs in parallel (same imgsz in model).
_DECOUPLE_PREVIEW_INFER = is_raspberry_pi()

# How many frames to skip between inference calls
# 0 = every frame, 1 = every other frame, etc.
# Pi: higher value → fewer ORT runs → more CPU for capture/letterbox (still imgsz 640 in model).
INFER_EVERY_N = 4 if is_raspberry_pi() else 1

# V4L2 grab warm-up passes (each grab drops a buffered frame; Pi pays a big latency tax).
_GRAB_WARMUP = 1 if is_raspberry_pi() else 2

# Pi capture resolution — model still letterboxes to imgsz (640); smaller source = less memcpy/resize work.
_PI_CAP_W, _PI_CAP_H = 512, 384

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
        self._layout_after_id: str | None = None
        self._fps_ui_counter = 0
        self._fps_ui_accum = 0.0

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
        right_w = int(UI.get("results_w", 260))
        right = tk.Frame(self._body, bg=COLORS["bg"], width=right_w)
        col_gap = int(UI["pad_y"])
        # Pack the fixed-width column first so it is not pushed off-screen.
        right.pack(side="right", fill="y", padx=(col_gap, 0))
        right.pack_propagate(False)
        left.pack(side="left", fill="both", expand=True)

        self._build_feed(left, vertical=False)
        self._build_results(right)
        self._build_controls(right)

        self._body.bind("<Configure>", self._on_body_configure)
        self.after_idle(self._apply_feed_fit)

    def _on_body_configure(self, event: tk.Event) -> None:
        if event.widget is not self._body:
            return
        if self._layout_after_id is not None:
            try:
                self.after_cancel(self._layout_after_id)
            except tk.TclError:
                pass
        self._layout_after_id = self.after(80, self._apply_feed_fit)

    def _apply_feed_fit(self) -> None:
        """Shrink the preview to whatever horizontal space is left for the camera column."""
        self._layout_after_id = None
        if self._feed_frame is None or self._body is None:
            return
        try:
            body_w = int(self._body.winfo_width())
        except tk.TclError:
            return
        if body_w < 80:
            return

        rw = int(UI.get("results_w", 260))
        gap = int(UI["pad_y"])
        avail = body_w - rw - gap - 8
        ideal_w = max(1, int(UI.get("feed_w", 860)))
        ideal_h = max(1, int(UI.get("feed_h", 540)))

        # Never wider than the space left of the results column (min 80 for tiny windows).
        new_w = max(80, min(ideal_w, avail))
        new_h = max(100, int(round(ideal_h * new_w / ideal_w)))

        if new_w == self._feed_target_w and new_h == self._feed_target_h:
            return
        self._feed_target_w = new_w
        self._feed_target_h = new_h
        try:
            self._feed_frame.config(width=new_w, height=new_h)
        except tk.TclError:
            return

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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _PI_CAP_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _PI_CAP_H)
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
        self._fps_ui_counter = 0
        self._fps_ui_accum = 0.0
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
            self._last_detections = []

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
                # Drop buffered frames when possible (Pi: single grab — cheaper).
                for _ in range(_GRAB_WARMUP):
                    if not self._cap.grab():
                        break
                ret, frame = self._cap.retrieve()
            except Exception:
                ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_n += 1
            tw, th = self._feed_target_w, self._feed_target_h

            if _DECOUPLE_PREVIEW_INFER:
                # Preview every frame; boxes use last ONNX results (mapped into letterboxed space).
                pil = _preview_from_bgr(frame, tw, th)
                with self._lock:
                    dets_snapshot = list(self._last_detections)
                if self.detector.is_loaded() and dets_snapshot:
                    oh, ow = frame.shape[0], frame.shape[1]
                    mapped = _map_dets_to_letterbox(dets_snapshot, ow, oh, tw, th)
                    pil = self.detector.draw_boxes(pil, mapped)
                with self._lock:
                    self._rendered_pil = pil
                if frame_n % (INFER_EVERY_N + 1) == 0 and not self._infer_busy:
                    self._infer_busy = True
                    threading.Thread(
                        target=self._infer_worker,
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()
            else:
                # Laptop / desktop: original fused path (preview skips some infer frames).
                if frame_n % (INFER_EVERY_N + 1) == 0 and not self._infer_busy:
                    self._infer_busy = True
                    threading.Thread(
                        target=self._infer_worker,
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()
                else:
                    pil = _preview_from_bgr(frame, tw, th)
                    if DRAW_STALE_BOXES and self._last_detections and self.detector.is_loaded():
                        pil = self.detector.draw_boxes(pil, self._last_detections)
                    with self._lock:
                        self._rendered_pil = pil

            # FPS (Pi: average + throttle Tk.after — main-thread scheduling was stealing cycles)
            now = time.perf_counter()
            elapsed = now - t_last
            t_last = now
            if is_raspberry_pi():
                self._fps_ui_counter += 1
                self._fps_ui_accum += elapsed
                if self._fps_ui_counter >= 8:
                    avg = self._fps_ui_accum / self._fps_ui_counter
                    fps = 1.0 / avg if avg > 0 else 0.0
                    self.after(0, self._fps_var.set, f"Camera: {fps:.1f} fps")
                    self._fps_ui_counter = 0
                    self._fps_ui_accum = 0.0
            else:
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self.after(0, self._fps_var.set, f"Camera: {fps:.1f} fps")

    # ------------------------------------------------------------------
    # Inference (runs in its own thread per invocation)
    # ------------------------------------------------------------------
    def _infer_worker(self, bgr_frame):
        if _DECOUPLE_PREVIEW_INFER:
            self._infer_decoupled(bgr_frame)
        else:
            self._infer_coupled(bgr_frame)

    def _infer_decoupled(self, bgr_frame):
        """ONNX only — capture thread owns _rendered_pil (smooth preview on Pi)."""
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            if self.detector.is_loaded():
                dets = self.detector.predict_pil(pil)
                with self._lock:
                    self._last_detections = list(dets)
                self.after(0, self._update_results_ui, dets)
            else:
                with self._lock:
                    self._last_detections = []
                self.after(0, self._update_results_ui, [])
        except Exception as exc:
            print(f"[CameraPanel] inference error: {exc}")
        finally:
            self._infer_busy = False

    def _infer_coupled(self, bgr_frame):
        """Legacy: draw on full frame then letterbox to preview (non-Pi)."""
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
        if self._layout_after_id is not None:
            try:
                self.after_cancel(self._layout_after_id)
            except tk.TclError:
                pass
            self._layout_after_id = None
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


def _map_dets_to_letterbox(
    dets: list[dict],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> list[dict]:
    """
    Map detection bboxes from camera frame pixels into letterboxed preview pixels
    (same geometry as _letterbox_bgr / _preview_from_bgr).
    """
    if not dets or orig_w < 1 or orig_h < 1:
        return []
    scale = min(target_w / orig_w, target_h / orig_h)
    nw, nh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
    left = (target_w - nw) // 2
    top = (target_h - nh) // 2
    sx = nw / orig_w
    sy = nh / orig_h
    out: list[dict] = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        out.append(
            {
                **d,
                "bbox": (
                    int(round(x1 * sx + left)),
                    int(round(y1 * sy + top)),
                    int(round(x2 * sx + left)),
                    int(round(y2 * sy + top)),
                ),
            }
        )
    return out


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
