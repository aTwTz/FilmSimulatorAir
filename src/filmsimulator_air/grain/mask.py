from __future__ import annotations

import numpy as np

from .field import _convolve_same


def _box_blur(x: np.ndarray, radius: int, axis: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = np.full((2 * radius + 1,), 1.0 / (2 * radius + 1), dtype=np.float32)
    return _convolve_same(x, kernel, axis)


def build_auto_mask(
    linear_gray: np.ndarray,
    diameter_px: float,
    *,
    texture_ref: float = 0.055,
    detail_floor: float = 0.48,
    flat_boost: float = 1.22,
    highlight_cut: float = 0.90,
    highlight_floor: float = 0.34,
) -> np.ndarray:
    gray = np.asarray(linear_gray, dtype=np.float32)
    radius = max(3, int(round(diameter_px)))
    mean = _box_blur(gray, radius, axis=0)
    mean = _box_blur(mean, radius, axis=1)
    mean_sq = _box_blur(gray * gray, radius, axis=0)
    mean_sq = _box_blur(mean_sq, radius, axis=1)
    var = np.clip(mean_sq - mean * mean, 0.0, None)

    texture = np.sqrt(var) / (mean + 1e-4)
    texture = texture / (texture + texture_ref)

    detail_gain = detail_floor + (1.0 - detail_floor) * (1.0 - texture)

    flat = 1.0 - texture
    boost = 1.0 + (flat_boost - 1.0) * np.clip(flat * 1.8 - 0.45, 0.0, 1.0)

    highlight = np.clip((highlight_cut - gray) / max(highlight_cut - 0.25, 1e-3), 0.0, 1.0)
    highlight = highlight_floor + (1.0 - highlight_floor) * highlight

    mask = detail_gain * boost * highlight
    return np.clip(mask, 0.0, 1.5).astype(np.float32)
