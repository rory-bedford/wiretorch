"""The wiretorch colour scheme, and Matplotlib themes in light and dark.

The scheme: background ``#263640`` (slate), primary ``#5096c0`` (blue), accents
``#2a9d8f`` (teal) and ``#e9c46a`` (yellow), highlight ``#e87878`` (red). Excitatory
neurons are red and inhibitory ones blue.

:func:`use_theme` styles Matplotlib with it. In a notebook it also makes every figure
display in both themes, each shown where it fits: in the documentation, the one that
matches the page; in Jupyter, the one that matches the system's light or dark mode.
The light figure is drawn as usual; the dark one is the same figure with its
background, ink and furniture swapped for their dark counterparts.
"""

import base64
import io

import numpy as np

SLATE = "#263640"
BLUE = "#5096c0"
TEAL = "#2a9d8f"
YELLOW = "#e9c46a"
RED = "#e87878"

EXCITATORY = RED
INHIBITORY = BLUE

#: Per theme: the background, the ink (text, labels), muted furniture (axes, ticks),
#: rules (grid lines), and the yellow, which is deepened on white to stay visible.
LIGHT = {
    "background": "#ffffff",
    "ink": SLATE,
    "muted": "#71838f",
    "rule": "#e4e9ec",
    "yellow": "#d9a83a",
}
DARK = {
    "background": "#131416",  # Furo's dark background, so figures blend into the docs
    "ink": "#d6d8da",
    "muted": "#8f979e",
    "rule": "#2a2e33",
    "yellow": YELLOW,
}

#: Ink and muted furniture of the light theme, for drawing with: in the dark theme
#: they become their dark counterparts (as does black).
INK = LIGHT["ink"]
MUTED = LIGHT["muted"]

_FONTS = ["Lato", "Open Sans", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def palette(mode: str = "light") -> list[str]:
    """The colour cycle: blue, teal, red, yellow, then lighter blue and teal."""
    colours = LIGHT if mode == "light" else DARK
    return [BLUE, TEAL, RED, colours["yellow"], "#8dbbd9", "#7cc4ba"]


def rc_params(mode: str = "light") -> dict:
    """Matplotlib rcParams for the theme ``"light"`` or ``"dark"``."""
    from cycler import cycler

    c = LIGHT if mode == "light" else DARK
    return {
        "figure.facecolor": c["background"],
        "figure.dpi": 110,
        "savefig.facecolor": c["background"],
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.facecolor": c["background"],
        "axes.edgecolor": c["muted"],
        "axes.labelcolor": c["ink"],
        "axes.titlecolor": c["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=palette(mode)),
        "grid.color": c["rule"],
        "grid.linewidth": 0.8,
        "text.color": c["ink"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelcolor": c["ink"],
        "ytick.labelcolor": c["ink"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": _FONTS,
        "mathtext.fontset": "dejavusans",
        "image.cmap": "wiretorch",
    }


def colormap():
    """A sequential colormap from slate through blue and teal to yellow."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "wiretorch", [SLATE, "#2f5f7e", BLUE, TEAL, YELLOW]
    )


def use_theme(mode: str = "light", *, notebook: bool = True) -> None:
    """Style Matplotlib with the wiretorch theme.

    Args:
        mode: ``"light"`` or ``"dark"``: the theme figures are drawn in.
        notebook: In IPython, display every figure in both themes (see the module
            docstring). Has no effect outside IPython.
    """
    import matplotlib as mpl

    if "wiretorch" not in mpl.colormaps:
        mpl.colormaps.register(colormap())
    mpl.rcParams.update(rc_params(mode))
    if notebook:
        _register_notebook_display(mode)


# ── Showing a figure in both themes ──────────────────────────────────────────


def _rgb(colour) -> tuple | None:
    from matplotlib.colors import to_rgba

    try:
        return tuple(np.round(to_rgba(colour)[:3], 3))
    except (TypeError, ValueError):
        return None


def _swaps(source: dict, target: dict) -> dict:
    """RGB of each themed colour in ``source`` -> its counterpart in ``target``."""
    swaps = {_rgb(source[k]): target[k] for k in source}
    if source is LIGHT:  # Matplotlib's own black and white, used explicitly
        swaps.setdefault(_rgb("black"), target["ink"])
    return swaps


def _swap(colour, swaps: dict):
    from matplotlib.colors import to_rgba

    rgb = _rgb(colour)
    if rgb not in swaps:
        return colour
    alpha = to_rgba(colour)[3]
    return to_rgba(swaps[rgb], alpha)


def _swap_array(colours, swaps: dict):
    colours = np.asarray(colours)
    if colours.ndim != 2 or len(colours) == 0:
        return colours
    return np.array([_swap(tuple(c), swaps) for c in colours])


def _recolour(fig, swaps: dict) -> None:
    """Swap every themed colour in ``fig`` for its counterpart."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    for artist in fig.findobj():
        if isinstance(artist, Text):
            artist.set_color(_swap(artist.get_color(), swaps))
        elif isinstance(artist, Line2D):
            artist.set_color(_swap(artist.get_color(), swaps))
            artist.set_markerfacecolor(_swap(artist.get_markerfacecolor(), swaps))
            artist.set_markeredgecolor(_swap(artist.get_markeredgecolor(), swaps))
        elif isinstance(artist, Patch):
            artist.set_facecolor(_swap(artist.get_facecolor(), swaps))
            artist.set_edgecolor(_swap(artist.get_edgecolor(), swaps))
        elif isinstance(artist, Collection):
            artist.set_facecolors(_swap_array(artist.get_facecolors(), swaps))
            artist.set_edgecolors(_swap_array(artist.get_edgecolors(), swaps))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis.get_major_ticks()
            if ticks:
                axis.set_tick_params(
                    which="both",
                    color=_swap(ticks[0].tick1line.get_color(), swaps),
                    labelcolor=_swap(ticks[0].label1.get_color(), swaps),
                )


def _png(fig) -> tuple[str, int]:
    import matplotlib as mpl

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    data = buffer.getvalue()
    width = int.from_bytes(data[16:20], "big")  # from the PNG header
    css_width = round(width * 100 / mpl.rcParams["savefig.dpi"])
    return base64.b64encode(data).decode(), css_width


_STYLE = (
    "<style>.wt-figure img{max-width:100%;height:auto}"
    ".wt-figure .wt-dark{display:none}"
    "@media (prefers-color-scheme: dark){.wt-figure .wt-light{display:none}"
    ".wt-figure .wt-dark{display:inline}}</style>"
)


def figure_html(fig, mode: str = "light") -> str:
    """HTML showing ``fig`` (drawn in theme ``mode``) in both themes."""
    drawn, other = (LIGHT, DARK) if mode == "light" else (DARK, LIGHT)
    for ax in fig.axes:
        if ax.images:  # grid lines would be drawn over the image
            ax.grid(False)
    images = {mode: _png(fig)}
    _recolour(fig, _swaps(drawn, other))
    images["dark" if mode == "light" else "light"] = _png(fig)
    _recolour(fig, _swaps(other, drawn))
    tags = "".join(
        f'<img class="wt-{name}" width="{width}" src="data:image/png;base64,{data}">'
        for name, (data, width) in sorted(images.items(), reverse=True)
    )
    return f'{_STYLE}<div class="wt-figure">{tags}</div>'


def _register_notebook_display(mode: str) -> None:
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # The inline backend starts with the first figure and then clears every figure
    # formatter, so start it now, and register again before each cell in case it is
    # restarted (``%matplotlib inline``).
    plt.close(plt.figure())

    def register(*_):
        formatters = shell.display_formatter.formatters
        formatters["text/html"].for_type(Figure, lambda fig: figure_html(fig, mode))
        # The HTML carries both images, so the plain PNG would only duplicate them.
        formatters["image/png"].type_printers.pop(Figure, None)

    register()
    global _registered
    if _registered is not None:
        shell.events.unregister("pre_run_cell", _registered)
    shell.events.register("pre_run_cell", register)
    _registered = register


_registered = None
