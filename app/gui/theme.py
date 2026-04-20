"""
app/gui/theme.py
Centralized color palette and font definitions.
Compatible with Raspberry Pi (DejaVu) and Windows (Segoe UI / Arial).
"""

import sys
import platform

# ----- Detect platform for font selection -----
_LINUX = platform.system() == "Linux"

def _f(size: int, weight: str = "normal") -> tuple:
    family = "DejaVu Sans" if _LINUX else "Segoe UI"
    return (family, size, weight)

_BASE_FONT_SIZES: dict[str, int] = {
    "h1": 20,
    "h2": 14,
    "body": 11,
    "label": 11,
    "small": 9,
}

def apply_font_scale(scale: float) -> None:
    """Mutates `FONTS` in-place so already-imported modules see updates."""
    scale = max(0.85, min(1.6, float(scale)))
    FONTS.update({
        "h1": _f(int(round(_BASE_FONT_SIZES["h1"] * scale)), "bold"),
        "h2": _f(int(round(_BASE_FONT_SIZES["h2"] * scale)), "bold"),
        "body": _f(int(round(_BASE_FONT_SIZES["body"] * scale))),
        "label": _f(int(round(_BASE_FONT_SIZES["label"] * scale)), "bold"),
        "small": _f(int(round(_BASE_FONT_SIZES["small"] * scale))),
        "big": _f(int(round(26 * scale)), "bold"),
        "result": _f(int(round(32 * scale)), "bold"),
        "mono": ("DejaVu Sans Mono" if _LINUX else "Courier New", int(round(10 * scale))),
    })

FONTS: dict[str, tuple] = {}
apply_font_scale(1.0)

UI: dict[str, int | bool] = {
    # Layout profile (set from main.py based on the actual screen)
    "is_small_screen": False,

    # Global padding rhythm
    "pad_x": 32,
    "pad_top": 28,
    "pad_y": 16,

    # Sidebar widths
    "sidebar_w": 200,
    "results_w": 260,

    # Camera feed render target (letterbox)
    "feed_w": 860,
    "feed_h": 540,

    # Text wrap lengths
    "settings_wrap": 720,
    "model_note_wrap": 230,

    # Touch targets
    "btn_padx": 18,
    "btn_pady": 10,
    "input_pady": 10,

    # Card rhythm (used by shared ui_components)
    "card_pad_x": 14,
    "card_pad_top": 14,
    "card_title_gap": 8,
}

def configure_ui_for_screen(screen_w: int, screen_h: int, *, on_pi: bool) -> None:
    """
    Configure UI metrics for a given display.

    Intended for Raspberry Pi LCDs (~800×480 / **1024×600** 7.5" class panels) to keep
    the UI readable without the layout getting squeezed.
    """
    sw, sh = int(screen_w), int(screen_h)
    small = (sw <= 1024 and sh <= 600) or (sw <= 900 or sh <= 520)
    UI["is_small_screen"] = bool(small)

    # Common 7.5" LCD native resolution (allow minor framebuffer variance)
    is_lcd_1024x600 = (1000 <= sw <= 1048) and (576 <= sh <= 624)

    if small:
        # Touch-friendly text, but tighter spacing to fit.
        apply_font_scale(1.15 if not on_pi else 1.2)
        if is_lcd_1024x600:
            # Usable content width ≈ 1024 − sidebar − horizontal padding
            UI.update({
                "pad_x": 14,
                "pad_top": 12,
                "pad_y": 8,
                "sidebar_w": 158,
                # Keep the right column tappable in horizontal mode.
                "results_w": 270,
                # Letterbox target for camera pipeline (~full content width, ~max height)
                # Keep height modest so stacked results fit on 600px displays.
                "feed_w": 640,
                "feed_h": 320,
                "settings_wrap": 780,
                "model_note_wrap": 560,
                "btn_padx": 18,
                "btn_pady": 12,
                "input_pady": 12,
                "card_pad_x": 14,
                "card_pad_top": 14,
                "card_title_gap": 8,
            })
        else:
            # Smaller panels (e.g. ~800×480)
            UI.update({
                "pad_x": 16,
                "pad_top": 14,
                "pad_y": 10,
                "sidebar_w": 170,
                # Right column must remain usable for touch.
                "results_w": 260,
                "feed_w": 560,
                "feed_h": 320,
                "settings_wrap": 520,
                "model_note_wrap": 320,
                "btn_padx": 16,
                "btn_pady": 11,
                "input_pady": 11,
                "card_pad_x": 14,
                "card_pad_top": 14,
                "card_title_gap": 8,
            })
    else:
        apply_font_scale(1.0)
        UI.update({
            "pad_x": 32,
            "pad_top": 28,
            "pad_y": 16,
            "sidebar_w": 200,
            "results_w": 260,
            "feed_w": 860,
            "feed_h": 540,
            "settings_wrap": 720,
            "model_note_wrap": 230,
            "btn_padx": 18,
            "btn_pady": 10,
            "input_pady": 10,
            "card_pad_x": 14,
            "card_pad_top": 14,
            "card_title_gap": 8,
        })

# ----- Dark theme palette -----
COLORS = {
    # Agriculture theme: deep green + near-black + high-contrast white
    "bg":            "#050807",   # near-black with green tint
    "card":          "#0B1411",   # dark evergreen panel
    "card_hover":    "#0F1C17",
    "border":        "#1F3A2F",   # muted green border
    "input":         "#0A1612",
    "text":          "#F4F7F6",   # off-white
    "muted":         "#A5B7AF",   # muted green-gray
    "accent":        "#16A34A",   # green-600
    "accent_hover":  "#15803D",   # green-700
    "sidebar":       "#040706",
    "sidebar_sel":   "#0B1411",
    "success":       "#22C55E",
    "success_hover": "#16A34A",
    "warning":       "#FACC15",   # yellow-400 (more legible on dark)
    "danger":        "#EF4444",
    "danger_hover":  "#DC2626",
    # class colours
    "mature":        "#22C55E",
    "immature":      "#FACC15",
    "damaged":       "#EF4444",
}
