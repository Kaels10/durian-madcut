"""
app/gui/ui_components.py
Shared UI component builders for consistent look & feel across panels.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from app.gui.theme import COLORS, FONTS, UI


def card(parent: tk.Widget, title: str, *, pad_bottom: int | None = None) -> tk.Frame:
    """
    Create a bordered card with a title and an inner body frame.
    Returns the body frame (place widgets inside it).
    """
    outer = tk.Frame(
        parent,
        bg=COLORS["card"],
        highlightthickness=1,
        highlightbackground=COLORS["border"],
    )
    outer.pack(fill="x", pady=(0, int(UI.get("pad_y", 16)) if pad_bottom is None else int(pad_bottom)))

    pad_x = int(UI.get("card_pad_x", 14))
    pad_top = int(UI.get("card_pad_top", 14))
    title_gap = int(UI.get("card_title_gap", 8))

    tk.Label(
        outer,
        text=title,
        font=FONTS["h2"],
        bg=COLORS["card"],
        fg=COLORS["text"],
    ).pack(anchor="w", padx=pad_x, pady=(pad_top, title_gap))

    body = tk.Frame(outer, bg=COLORS["card"])
    body.pack(fill="x", padx=pad_x, pady=(0, pad_top))
    return body


def primary_button(parent: tk.Widget, text: str, *, command=None) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        font=FONTS["h2"],
        bg=COLORS["accent"],
        fg="white",
        activebackground=COLORS["accent_hover"],
        activeforeground="white",
        relief="flat",
        padx=int(UI.get("btn_padx", 18)),
        pady=int(UI.get("btn_pady", 10)),
        cursor="hand2",
        command=command,
    )


def secondary_button(parent: tk.Widget, text: str, *, command=None) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        font=FONTS["h2"],
        bg=COLORS["card_hover"],
        fg=COLORS["text"],
        activebackground=COLORS["border"],
        activeforeground=COLORS["text"],
        relief="flat",
        padx=int(UI.get("btn_padx", 18)),
        pady=int(UI.get("btn_pady", 10)),
        cursor="hand2",
        command=command,
    )


def danger_button(parent: tk.Widget, text: str, *, command=None) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        font=FONTS["body"],
        bg=COLORS["danger"],
        fg="white",
        activebackground=COLORS["danger_hover"],
        activeforeground="white",
        relief="flat",
        padx=int(UI.get("btn_padx", 18)),
        pady=max(6, int(UI.get("btn_pady", 10)) - 2),
        cursor="hand2",
        command=command,
    )


def styled_entry(parent: tk.Widget, *, textvariable: tk.StringVar | None = None) -> tk.Entry:
    return tk.Entry(
        parent,
        textvariable=textvariable,
        font=FONTS["body"],
        bg=COLORS["input"],
        fg=COLORS["text"],
        insertbackground=COLORS["text"],
        relief="flat",
        highlightthickness=1,
        highlightcolor=COLORS["accent"],
        highlightbackground=COLORS["border"],
    )


def styled_scale(parent: tk.Widget, *, variable, from_: float, to: float, command=None) -> ttk.Scale:
    # ttk scale uses ttk theme; we keep it but rely on surrounding card styling.
    return ttk.Scale(parent, variable=variable, from_=from_, to=to, orient="horizontal", command=command)

