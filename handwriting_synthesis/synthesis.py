"""
handwriting_synthesis.synthesis
================================
Single public entry point:

    from handwriting_synthesis import generate

    strokes = generate("hello world", style=5, bias=0.75)

Returns a list of np.ndarray of shape [N, 2] — one array per pen-down
stroke segment, in absolute (x, y) pixel coordinates scaled to a
128-pixel-tall canvas.
"""

from __future__ import annotations

import importlib.resources
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# ── lazy singleton so TF is only loaded once ──────────────────────────────────
_lock = threading.Lock()
_hand = None


def _get_hand():
    global _hand
    if _hand is not None:
        return _hand
    with _lock:
        if _hand is not None:
            return _hand
        # Suppress TF spam
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from .hand import Hand
        _hand = Hand()
    return _hand


# ── public API ─────────────────────────────────────────────────────────────────

def generate(
    text: str,
    style: int = 0,
    bias: float = 0.75,
) -> list[np.ndarray]:
    """
    Generate handwriting strokes for *text*.

    Parameters
    ----------
    text  : str   — the text to render (keep under ~60 chars for best results)
    style : int   — writing style 0-9
    bias  : float — legibility bias; higher = cleaner/more legible (0.1–1.0)

    Returns
    -------
    List of np.ndarray, each shape [N, 2] (x, y absolute coordinates).
    The coordinate space is normalised so the line fits in a 128px-tall canvas.
    Consecutive arrays are separate pen-down strokes (lift pen between them).
    """
    if not 0 <= style <= 9:
        raise ValueError(f"style must be 0-9, got {style}")
    if not 0.0 < bias <= 1.0:
        raise ValueError(f"bias must be in (0, 1], got {bias}")

    hand = _get_hand()

    # _sample returns [T, 3]: dx, dy, pen_up
    raw = hand._sample([text], biases=[bias], styles=[style])
    if isinstance(raw, list):
        raw = raw[0]

    return _to_absolute_segments(raw)


def _to_absolute_segments(strokes_raw: np.ndarray) -> list[np.ndarray]:
    """Convert relative (dx, dy, pen_up) array to absolute stroke segments."""
    x, y = 0.0, 0.0
    current: list[tuple[float, float]] = []
    segments: list[np.ndarray] = []

    for dx, dy, pen_up in strokes_raw:
        x += float(dx)
        y += float(dy)
        current.append((x, y))
        if pen_up > 0.5 and len(current) >= 2:
            segments.append(np.array(current, dtype=np.float32))
            current = [(x, y)]

    if len(current) >= 2:
        segments.append(np.array(current, dtype=np.float32))

    return segments
