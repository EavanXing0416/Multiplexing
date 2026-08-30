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


def _with_bounds(
    cmap: LinearSegmentedColormap,
    *,
    under: tuple[float, float, float] | str | None = None,
    over: tuple[float, float, float] | str | None = None,
) -> LinearSegmentedColormap:
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)
    return cmap


def build_default_colormaps() -> dict[str, LinearSegmentedColormap]:
    main_data = _with_bounds(
        make_cmap(
            "main_data_fields",
            [
                (0.00, rgb255(255, 170, 30)),
                (0.10, rgb255(255, 205, 65)),
                (0.52, rgb255(235, 233, 228)),
                (0.90, rgb255(84, 222, 226)),
                (1.00, rgb255(72, 189, 237)),
            ],
            register=False,
        ),
        under=rgb255(255, 120, 20),
        over=rgb255(65, 185, 235),
    )

    performance_positive = _with_bounds(
        make_cmap(
            "performance_positive",
            [
                (0.00, rgb255(238, 0, 238)),
                (0.10, rgb255(198, 112, 236)),
                (0.36, rgb255(123, 49, 201)),
                (0.37, rgb255(238, 221, 110)),
                (0.64, rgb255(186, 184, 28)),
                (0.65, rgb255(128, 223, 214)),
                (0.90, rgb255(52, 211, 83)),
                (1.00, rgb255(31, 255, 20)),
            ],
            register=False,
        ),
        under=rgb255(239, 0, 239),
        over=rgb255(25, 255, 10),
    )

    performance_negative = _with_bounds(
        make_cmap(
            "performance_negative",
            [
                (0.00, rgb255(102, 146, 236)),
                (0.10, rgb255(161, 199, 240)),
                (0.36, rgb255(83, 162, 221)),
                (0.37, rgb255(201, 208, 192)),
                (0.64, rgb255(173, 177, 135)),
                (0.65, rgb255(247, 177, 146)),
                (0.90, rgb255(228, 61, 61)),
                (1.00, rgb255(255, 86, 86)),
            ],
            register=False,
        ),
        under=rgb255(95, 140, 235),
        over=rgb255(255, 85, 85),
    )

    data_properties = _with_bounds(
        make_cmap(
            "data_properties",
            [
                (0.00, rgb255(255, 238, 0)),
                (0.10, rgb255(214, 208, 83)),
                (0.52, rgb255(240, 232, 226)),
                (0.90, rgb255(147, 76, 236)),
                (1.00, rgb255(182, 116, 240)),
            ],
            register=False,
        ),
        under=rgb255(255, 235, 0),
        over=rgb255(185, 110, 245),
    )

    training_positive = _with_bounds(
        make_cmap(
            "training_signal_positive",
            [
                (0.00, rgb255(238, 0, 238)),
                (0.10, rgb255(203, 146, 243)),
                (0.36, rgb255(125, 165, 236)),
                (0.64, rgb255(208, 246, 87)),
                (0.90, rgb255(168, 232, 162)),
                (1.00, rgb255(29, 255, 10)),
            ],
            register=False,
        ),
        under=rgb255(239, 0, 239),
        over=rgb255(25, 255, 20),
    )

    training_negative = _with_bounds(
        make_cmap(
            "training_signal_negative",
            [
                (0.00, rgb255(102, 146, 236)),
                (0.18, rgb255(136, 220, 221)),
                (0.45, rgb255(211, 245, 117)),
                (0.72, rgb255(246, 180, 148)),
                (0.90, rgb255(255, 145, 145)),
                (1.00, rgb255(255, 95, 95)),
            ],
            register=False,
        ),
        under=rgb255(95, 140, 235),
        over=rgb255(255, 85, 85),
    )

    training_neutral = _with_bounds(
        make_cmap(
            "training_signal_neutral",
            [
                (0.00, rgb255(178, 191, 161)),
                (0.10, rgb255(216, 223, 229)),
                (0.50, rgb255(185, 183, 204)),
                (0.90, rgb255(176, 155, 108)),
                (1.00, rgb255(206, 192, 143)),
            ],
            register=False,
        ),
        under=rgb255(170, 186, 156),
        over=rgb255(204, 188, 142),
    )

    contour_glyph = _with_bounds(
        make_cmap(
            "contour_and_hollow_glyphs",
            [
                (0.00, rgb255(48, 92, 43)),
                (0.14, rgb255(54, 152, 133)),
                (0.28, rgb255(39, 119, 191)),
                (0.42, rgb255(65, 59, 214)),
                (0.56, rgb255(140, 18, 161)),
                (0.70, rgb255(190, 29, 105)),
                (0.84, rgb255(182, 73, 26)),
                (1.00, rgb255(165, 140, 71)),
            ],
            register=False,
        ),
        under=rgb255(47, 88, 41),
        over=rgb255(166, 139, 70),
    )

    sampled_points = _with_bounds(
        make_cmap(
            "sampled_data_points",
            [
                (0.00, rgb255(255, 255, 255)),
                (0.50, rgb255(255, 255, 255)),
                (0.5001, rgb255(0, 0, 0)),
                (1.00, rgb255(0, 0, 0)),
            ],
            register=False,
        ),
        under=rgb255(255, 255, 255),
        over=rgb255(0, 0, 0),
    )

    error = mpl.colormaps["Greys"].copy()
    error.set_under("white")
    error.set_over("black")

    return {
        "main_data": main_data,
        "main_data_zebra": make_zebra_cmap(main_data, -1.2, 1.2, levels=20, name="main_data_zebra"),
        "performance_positive": performance_positive,
        "performance_positive_zebra": make_zebra_cmap(
            performance_positive, 0.0, 1.0, levels=16, name="performance_positive_zebra"
        ),
        "performance_negative": performance_negative,
        "performance_negative_zebra": make_zebra_cmap(
            performance_negative, -1.0, 1.0, levels=16, name="performance_negative_zebra"
        ),
        "data_properties": data_properties,
        "data_properties_zebra": make_zebra_cmap(
            data_properties, -1.0, 1.0, levels=18, name="data_properties_zebra"
        ),
        "training_positive": training_positive,
        "training_positive_zebra": make_zebra_cmap(
            training_positive, 0.0, 1.0, levels=16, name="training_positive_zebra"
        ),
        "training_negative": training_negative,
        "training_negative_zebra": make_zebra_cmap(
            training_negative, -1.0, 1.0, levels=16, name="training_negative_zebra"
        ),
        "training_neutral": training_neutral,
        "training_neutral_zebra": make_zebra_cmap(
            training_neutral, -1.0, 1.0, levels=16, name="training_neutral_zebra"
        ),
        "contour_glyph": contour_glyph,
        "sampled_points": sampled_points,
        "error": error,
    }
