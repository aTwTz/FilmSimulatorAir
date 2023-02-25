from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .core import RenderOptions, options_from_args, render_image
from .profiles import PRESETS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _nonneg_float(value: str) -> float:
    x = float(value)
    if x < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def _positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="FilmSimulatorAir",
        description="Render photos with physical-ish film grain in density domain.",
    )
    p.add_argument("input", help="input image file or directory for batch mode")
    p.add_argument("output", help="output image file or directory for batch mode")
    p.add_argument("--preset", choices=sorted(PRESETS), help="film preset (default: custom options)")
    p.add_argument("--strength", type=_nonneg_float, default=None, help="grain strength multiplier")
    p.add_argument("--grain-um", type=_positive_float, default=None, help="visible grain cluster diameter in micrometers")
    p.add_argument("--dpi", type=int, default=None, help="output resolution used for um->px conversion; default 1200")
    p.add_argument("--gamma", type=_positive_float, default=None, help="density-to-noise exponent")
    p.add_argument("--color", action="store_true", default=None, help="color emulsion grain")
    p.add_argument("--mono", action="store_false", dest="color", help="monochrome film grain")
    p.add_argument("--auto", dest="auto_mask", action="store_true", default=None, help="enable content-adaptive mask")
    p.add_argument("--no-auto", dest="auto_mask", action="store_false", help="disable content-adaptive mask")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible grain")
    p.add_argument("--recursive", action="store_true", help="batch mode: recurse into input directory")
    p.add_argument("--format", default="png", choices=["png", "jpg", "webp", "tif"], help="batch output format")
    p.add_argument("--jobs", type=int, default=1, help="reserved for future parallel batching; kept for CLI stability")
    return p


def _iter_inputs(input_path: Path, recursive: bool):
    if input_path.is_file():
        yield input_path, input_path.name
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path, path.name if not recursive else str(path.relative_to(input_path))


def _options_from_namespace(ns: argparse.Namespace) -> RenderOptions:
    return options_from_args(
        ns.preset,
        strength=ns.strength,
        gamma=ns.gamma,
        grain_um=ns.grain_um,
        dpi=ns.dpi,
        color=ns.color,
        auto_mask=ns.auto_mask,
        seed=ns.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    input_path = Path(ns.input)
    output_path = Path(ns.output)
    if not input_path.exists():
        parser.error(f"input does not exist: {input_path}")
    if ns.jobs != 1:
        print("note: --jobs is reserved; batch mode is sequential in this build", file=sys.stderr)

    opt = _options_from_namespace(ns)
    files = list(_iter_inputs(input_path, ns.recursive))
    if not files:
        parser.error(f"no images found under: {input_path}")

    batch = input_path.is_dir() or output_path.is_dir() or len(files) > 1
    if batch:
        output_path.mkdir(parents=True, exist_ok=True)

    for idx, (src, rel_name) in enumerate(files, 1):
        if batch:
            dst = output_path / f"{Path(rel_name).stem}.{ns.format}"
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dst = output_path
        try:
            with Image.open(src) as im:
                rendered = render_image(im, opt)
            rendered.save(dst)
        except Exception as exc:
            print(f"error: failed on {src}: {exc}", file=sys.stderr)
            return 2
        prefix = f"[{idx}/{len(files)}] " if batch else ""
        print(f"{prefix}{src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
