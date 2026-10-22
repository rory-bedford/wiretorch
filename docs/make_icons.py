"""Make the documentation's logo and favicons from ``assets/wiretorch-logo.png``.

    uv run python docs/make_icons.py

Writes, to ``docs/_static/``: the logo cropped to its content (``logo.png``, also used
by the README), ``favicon.ico`` (16, 32 and 48 px) and PNG favicons (transparent, the
strokes thickened for the small sizes), an Apple touch icon
(on the slate background, since iOS fills transparency with black), Android icons and
the web manifest that lists them.
"""

import json
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "_static"
SLATE = (0x26, 0x36, 0x40, 255)


def square(image: Image.Image, padding: float) -> Image.Image:
    """``image`` centred on a transparent square, with ``padding`` (a fraction) around."""
    side = round(max(image.size) * (1 + 2 * padding))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2), image)
    return canvas


def icon(logo: Image.Image, size: int, *, bold: bool) -> Image.Image:
    """The logo on a transparent square, ``size`` px; ``bold`` thickens its strokes
    first, so they survive the smallest sizes."""
    work = 256
    mark = square(logo, padding=0.02).resize((work, work), Image.LANCZOS)
    if bold:
        mark = mark.filter(ImageFilter.MaxFilter(9))
    return mark.resize((size, size), Image.LANCZOS)


def main() -> None:
    logo = Image.open(ROOT / "assets" / "wiretorch-logo.png").convert("RGBA")
    logo = logo.crop(logo.getbbox())
    OUT.mkdir(exist_ok=True)

    wide = logo.copy()
    wide.thumbnail((800, 800), Image.LANCZOS)
    wide.save(OUT / "logo.png", optimize=True)

    small = {size: icon(logo, size, bold=True) for size in (16, 32, 48)}
    small[48].save(
        OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)],
        append_images=[small[16], small[32]],
    )  # fmt: skip
    for size in (16, 32):
        small[size].save(OUT / f"favicon-{size}.png")
    for size in (192, 512):
        icon(logo, size, bold=False).save(OUT / f"icon-{size}.png")

    touch = Image.new("RGBA", (180, 180), SLATE)
    inner = square(logo, padding=0.12).resize((180, 180), Image.LANCZOS)
    touch.alpha_composite(inner)
    touch.convert("RGB").save(OUT / "apple-touch-icon.png")

    manifest = {
        "name": "wiretorch",
        "short_name": "wiretorch",
        "icons": [
            {"src": f"icon-{s}.png", "sizes": f"{s}x{s}", "type": "image/png"}
            for s in (192, 512)
        ],
        "theme_color": "#263640",
        "background_color": "#263640",
        "display": "standalone",
    }
    (OUT / "site.webmanifest").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
