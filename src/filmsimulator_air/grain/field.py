from __future__ import annotations

import math

import numpy as np

MICRONS_PER_INCH = 25400.0


def diameter_px(grain_um: float, dpi: int) -> float:
    if grain_um <= 0:
        raise ValueError("grain_um must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    return float(grain_um) * float(dpi) / MICRONS_PER_INCH


def grain_sigma_px(grain_um: float, dpi: int) -> float:
    d = diameter_px(grain_um, dpi)
    return max(0.65, d * 0.35)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)
    return k


def _convolve_same(x: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.size // 2
    pads = [(pad, pad) if i == axis else (0, 0) for i in range(x.ndim)]
    xp = np.pad(x, pads, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, xp)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma)
    y = _convolve_same(x, k, axis=0)
    y = _convolve_same(y, k, axis=1)
    y /= math.sqrt(float(np.sum(k * k)))
    return y.astype(np.float32)


def generate_grain(
    shape: tuple[int, int] | tuple[int, int, int],
    grain_um: float,
    dpi: int,
    *,
    mono: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if mono:
        if len(shape) == 3:
            h, w, _ = shape
        else:
            h, w = shape
        noise = rng.standard_normal((h, w), dtype=np.float32)
        return _blur(noise, grain_sigma_px(grain_um, dpi))

    h, w, _ = shape
    sigmas = (grain_sigma_px(grain_um, dpi) * s for s in (1.00, 1.08, 0.94))
    channels = [
        _blur(rng.standard_normal((h, w), dtype=np.float32), sigma)
        for sigma in sigmas
    ]
    return np.stack(channels, axis=-1).astype(np.float32)
