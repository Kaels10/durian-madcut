"""
main.py – Durian Maturity Assessment App
Entry point. Run with:  python3 main.py
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Ensure app package is in path
sys.path.insert(0, str(Path(__file__).parent))

from app.gui.theme import COLORS, FONTS
from app.ml.detector import DurianDetector
from app.utils.model_autoload import try_autoload_on_startup


def _check_dependencies() -> bool:
    """Soft-check for key dependencies before opening the window."""
    missing = []
    try:
        import PIL
    except ImportError:
        missing.append("pillow")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python-headless")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    if missing:
        msg = (
            "Some dependencies are missing:\n\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nPlease run:  ./setup.sh  (or setup.bat on Windows)"
        )
        # Tkinter may not be available if really broken, so try both
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing dependencies", msg)
            root.destroy()
        except Exception:
            print(msg)
        return False
    return True


def main():
    if not _check_dependencies():
        sys.exit(1)

    from app.gui.main_window import MainWindow

    detector = DurianDetector()
    loaded_path = try_autoload_on_startup(detector)
    if loaded_path:
        print(f"[Durian] Model auto-loaded: {loaded_path}")

    root = tk.Tk()
    root.title("Durian Maturity Assessment")
    root.geometry("1100x700")
    root.minsize(900, 600)
    root.configure(bg=COLORS["bg"])

    # Window icon (graceful fallback)
    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if icon_path.exists():
        try:
            icon = tk.PhotoImage(file=str(icon_path))
            root.iconphoto(True, icon)
        except Exception:
            pass

    # Apply ttk theme globally
    try:
        from tkinter import ttk
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TProgressbar",
                        troughcolor=COLORS["card"],
                        background=COLORS["accent"],
                        bordercolor=COLORS["border"],
                        lightcolor=COLORS["accent"],
                        darkcolor=COLORS["accent"])
        style.configure("Vertical.TScrollbar",
                        background=COLORS["border"],
                        troughcolor=COLORS["card"],
                        arrowcolor=COLORS["muted"],
                        bordercolor=COLORS["card"])
    except Exception:
        pass

    app = MainWindow(root, detector)

    # Center on screen
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    root.mainloop()


if __name__ == "__main__":
    main()
