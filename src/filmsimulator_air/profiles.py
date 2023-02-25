from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilmProfile:
    name: str
    strength: float
    gamma: float
    grain_um: float
    color: bool
    dpi: int
    note: str = "approximate preset; tune with the compare matrix"


PRESETS: dict[str, FilmProfile] = {
    "portra400": FilmProfile("portra400", strength=0.52, gamma=0.52, grain_um=8.0, color=True, dpi=1200),
    "ektachrome": FilmProfile("ektachrome", strength=0.46, gamma=0.50, grain_um=7.0, color=True, dpi=1200),
    "hp5": FilmProfile("hp5", strength=0.82, gamma=0.55, grain_um=11.0, color=False, dpi=1200),
    "tri-x": FilmProfile("tri-x", strength=0.74, gamma=0.55, grain_um=10.0, color=False, dpi=1200),
    "delta3200": FilmProfile("delta3200", strength=1.18, gamma=0.62, grain_um=15.0, color=False, dpi=1200),
    "cinestill800": FilmProfile("cinestill800", strength=0.96, gamma=0.58, grain_um=13.0, color=True, dpi=1200),
}


def get_profile(name: str) -> FilmProfile:
    try:
        return PRESETS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown preset {name!r}; choose one of: {choices}") from exc
