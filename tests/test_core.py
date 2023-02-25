import unittest

import numpy as np

from filmsimulator_air.color import density_to_rgb, linear_to_srgb, rgb_to_density, srgb_to_linear
from filmsimulator_air.core import RenderOptions, render_array
from filmsimulator_air.grain.field import diameter_px, grain_sigma_px
from filmsimulator_air.profiles import PRESETS, get_profile


class ColorTests(unittest.TestCase):
    def test_srgb_linear_roundtrip(self):
        x = np.linspace(0, 1, 257, dtype=np.float32)
        y = linear_to_srgb(srgb_to_linear(x))
        np.testing.assert_allclose(x, y, atol=2e-3)

    def test_density_roundtrip(self):
        lin = np.array([[[1.0, 0.5, 0.05], [0.2, 0.01, 0.001]]], dtype=np.float32)
        np.testing.assert_allclose(density_to_rgb(rgb_to_density(lin)), lin, rtol=1e-3, atol=1e-5)


class GrainTests(unittest.TestCase):
    def test_micron_to_px(self):
        self.assertAlmostEqual(diameter_px(25400, 1), 1.0)
        self.assertGreaterEqual(grain_sigma_px(10, 1200), 0.65)

    def test_render_changes_image_and_clips(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        opt = RenderOptions(strength=1.0, grain_um=12, dpi=900, seed=123, color=False)
        out = render_array(base, opt)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, base.shape)
        self.assertTrue(np.any(out != base))
        self.assertTrue(np.all(out <= 255))

    def test_seed_reproducible(self):
        img = np.full((64, 64, 3), 120, dtype=np.uint8)
        a = render_array(img, RenderOptions(seed=7))
        b = render_array(img, RenderOptions(seed=7))
        np.testing.assert_array_equal(a, b)

    def test_presets_exist(self):
        for name in ("hp5", "portra400"):
            self.assertIn(name, PRESETS)
        self.assertFalse(get_profile("HP5").color)


if __name__ == "__main__":
    unittest.main()
