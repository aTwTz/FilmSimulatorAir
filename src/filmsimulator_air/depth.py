from __future__ import annotations

import numpy as np


def estimate_depth(image_rgb_u8: np.ndarray) -> np.ndarray | None:
    try:
        import torch
    except Exception:
        return None
    return None
