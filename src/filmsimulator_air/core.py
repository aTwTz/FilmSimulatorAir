from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .color import (
    density_to_rgb,
    linear_luminance,
    linear_to_srgb,
    normalize_density,
    rgb_to_density,
    srgb_to_linear,
)
from .grain.field import diameter_px, generate_grain
from .grain.mask import build_auto_mask
from .profiles import FilmProfile, get_profile


@dataclass(frozen=True)
class RenderOptions:
    strength: float = 0.8
    gamma: float = 0.55
    grain_um: float = 10.0
    dpi: int = 1200
    color: bool = True
    auto_mask: bool = True
    seed: int | None = None
    density_base: float = 0.004
    density_amp: float = 0.115
    depth_weight: float = 0.18

    @classmethod
    def from_profile(cls, profile: FilmProfile, **overrides) -> "RenderOptions":
        values = {
            "strength": profile.strength,
            "gamma": profile.gamma,
            "grain_um": profile.grain_um,
            "dpi": profile.dpi,
            "color": profile.color,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def _sigma_density(norm_density: np.ndarray, opt: RenderOptions) -> np.ndarray:
    return (opt.strength * (opt.density_base + opt.density_amp * np.power(norm_density, opt.gamma))).astype(np.float32)


def render_array(
    rgb_u8: np.ndarray,
    opt: RenderOptions,
    *,
    depth_map: np.ndarray | None = None,
) -> np.ndarray:
    if rgb_u8.ndim != 3 or rgb_u8.shape[-1] != 3:
        raise ValueError("rgb_u8 must be an HxWx3 uint8 array")
    h, w, _ = rgb_u8.shape
    rng = np.random.default_rng(opt.seed)

    srgb = rgb_u8.astype(np.float32) / 255.0
    linear = srgb_to_linear(srgb)

    if opt.color:
        density = rgb_to_density(linear)
        norm = normalize_density(density)
        sigma = _sigma_density(norm, opt)
        grain = generate_grain((h, w, 3), opt.grain_um, opt.dpi, mono=False, rng=rng)
        gray = linear_luminance(linear)
    else:
        gray = linear_luminance(linear)
        density = rgb_to_density(gray)[..., None]
        norm = normalize_density(density)
        sigma = _sigma_density(norm, opt)
        grain = generate_grain((h, w, 3), opt.grain_um, opt.dpi, mono=True, rng=rng)[..., None]

    if opt.auto_mask:
        mask = build_auto_mask(gray, diameter_px(opt.grain_um, opt.dpi))
    else:
        mask = np.ones((h, w), dtype=np.float32)
    mask = mask[..., None] if sigma.ndim == 3 else mask

    if depth_map is not None:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.shape != (h, w):
            raise ValueError("depth_map must be HxW")
        depth = np.clip(depth, 0.0, 1.0)
        depth_gain = 1.0 + opt.depth_weight * (depth - 0.5)
        mask = mask * depth_gain[..., None if sigma.ndim == 3 else None]

    noisy_density = density + sigma * mask * grain
    out_linear = density_to_rgb(noisy_density)
    if not opt.color:
        out_linear = np.repeat(out_linear, 3, axis=-1)
    out_srgb = linear_to_srgb(np.clip(out_linear, 0.0, 1.0))
    return np.round(out_srgb * 255.0).astype(np.uint8)


def render_image(image: Image.Image, opt: RenderOptions) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    out = render_array(arr, opt)
    return Image.fromarray(out, mode="RGB")


def options_from_args(preset: str | None, **overrides) -> RenderOptions:
    if preset:
        profile = get_profile(preset)
        return RenderOptions.from_profile(profile, **overrides)
    return RenderOptions(**{k: v for k, v in overrides.items() if v is not None})
