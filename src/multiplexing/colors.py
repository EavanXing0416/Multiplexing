from __future__ import annotations

import colorsys

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def rgb255(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    h_norm = (h * 1.42857 - 5.7) / 360.0
    s_norm = s / 255.0
    l_norm = l / 255.0
    return colorsys.hls_to_rgb(h_norm, l_norm, s_norm)


def make_cmap(
    name: str,
    control_points: list[tuple[float, tuple[float, float, float]]],
    *,
    register: bool = True,
) -> LinearSegmentedColormap:
    cdict = {"red": [], "green": [], "blue": []}
    for pos, (r, g, b) in control_points:
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))
    cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    if register:
        try:
            mpl.colormaps.register(cmap)
        except ValueError:
            pass
    return cmap


def make_zebra_cmap(
    base_cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    *,
    levels: int = 20,
    name: str | None = None,
    white_color: tuple[float, float, float, float] = (1, 1, 1, 0),
) -> LinearSegmentedColormap:
    xs = np.linspace(0, 1, 256)
    vals = vmin + xs * (vmax - vmin)
    rgba = base_cmap(xs)
    level_edges = np.linspace(vmin, vmax, levels)
    mask_intervals = [(level_edges[i], level_edges[i + 1]) for i in range(0, len(level_edges) - 1, 2)]

    for low, high in mask_intervals:
        mask = (vals >= low) & (vals <= high)
        rgba[mask] = white_color

    cmap_name = name or f"{base_cmap.name}_zebra"
    cmap = LinearSegmentedColormap.from_list(cmap_name, rgba)
    if getattr(base_cmap, "_rgba_under", None) is not None:
        cmap.set_under(base_cmap._rgba_under)
    if getattr(base_cmap, "_rgba_over", None) is not None:
        cmap.set_over(base_cmap._rgba_over)
    return cmap


def build_default_colormaps() -> dict[str, LinearSegmentedColormap]:
    data_cmap = make_cmap(
        "multiplexing_data",
        [
            (0.00, hsl_to_rgb(30, 255, 160)),
            (0.50, rgb255(190, 190, 190)),
            (1.00, hsl_to_rgb(130, 255, 160)),
        ],
        register=False,
    )
    data_cmap.set_under(hsl_to_rgb(20, 255, 140))
    data_cmap.set_over(hsl_to_rgb(140, 255, 140))

    phy_cmap = make_cmap(
        "multiplexing_phy",
        [
            (0.00, hsl_to_rgb(160, 180, 180)),
            (0.33, hsl_to_rgb(140, 220, 220)),
            (0.33001, hsl_to_rgb(120, 60, 180)),
            (0.66, hsl_to_rgb(60, 60, 200)),
            (0.66001, hsl_to_rgb(30, 160, 160)),
            (1.00, hsl_to_rgb(5, 220, 220)),
        ],
        register=False,
    )
    phy_cmap.set_under(rgb255(170, 255, 90))
    phy_cmap.set_over(rgb255(0, 255, 80))

    error_cmap = mpl.colormaps["Greys"].copy()
    error_cmap.set_under("white")
    error_cmap.set_over("black")

    return {
        "data": data_cmap,
        "data_zebra": make_zebra_cmap(data_cmap, -1.2, 1.2, levels=20),
        "phy": phy_cmap,
        "phy_zebra": make_zebra_cmap(phy_cmap, 0.0, 0.24, levels=16),
        "error": error_cmap,
    }
