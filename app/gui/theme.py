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

FONTS = {
    "h1":    _f(20, "bold"),
    "h2":    _f(14, "bold"),
    "body":  _f(11),
    "label": _f(11, "bold"),
    "small": _f(9),
    "mono":  ("DejaVu Sans Mono" if _LINUX else "Courier New", 10),
}

# ----- Dark theme palette -----
COLORS = {
    "bg":            "#0F172A",   # slate-900
    "card":          "#1E293B",   # slate-800
    "card_hover":    "#263349",
    "border":        "#334155",   # slate-700
    "input":         "#1E293B",
    "text":          "#F1F5F9",   # slate-100
    "muted":         "#94A3B8",   # slate-400
    "accent":        "#22C55E",   # green-500
    "accent_hover":  "#16A34A",   # green-600
    "sidebar":       "#0D1927",
    "sidebar_sel":   "#1E293B",
    "success":       "#22C55E",
    "success_hover": "#16A34A",
    "warning":       "#EAB308",
    "danger":        "#EF4444",
    "danger_hover":  "#DC2626",
    # class colours
    "mature":        "#22C55E",
    "immature":      "#EAB308",
    "damaged":       "#EF4444",
}
