from __future__ import annotations

import numpy as np

EPS = 1e-6
DENSITY_SCALE = 3.0


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def rgb_to_density(linear_rgb: np.ndarray) -> np.ndarray:
    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=np.float32), EPS, 1.0)
    return -np.log10(linear_rgb).astype(np.float32)


def density_to_rgb(density: np.ndarray) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=np.float32), 0.0, DENSITY_SCALE * 2.0)
    return np.power(10.0, -density).astype(np.float32)


def linear_luminance(linear_rgb: np.ndarray) -> np.ndarray:
    r, g, b = np.moveaxis(linear_rgb, -1, 0)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)


def normalize_density(density: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(density, dtype=np.float32) / DENSITY_SCALE, 0.0, 1.0).astype(np.float32)
