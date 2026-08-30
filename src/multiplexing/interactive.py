from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import textwrap

import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

from .colors import build_default_colormaps, make_zebra_cmap
from .io import filter_rows, load_numeric_csv
from .plots import _draw_radar_glyph, _draw_ring_glyph, _normalize_metric, draw_glyph_legend, grid_scatter, reshape_regular_grid


SPATIAL_FIELDS = {
    "testing": ["u_referance", "u_pred", "error"],
    "training": ["u_ref", "u_pred", "phy_loss", "data_loss", "phys_eff", "bc_loss", "u_pred_grad", "phy_loss_grad", "entk"],
}

VIS_OPTIONS = {
    "testing": ["heatmap", "zebra_map", "contour", "scatter"],
    "training": ["heatmap", "zebra_map", "contour", "scatter", "glyph"],
}

COLOR_STYLE_OPTIONS = [
    "main_data",
    "performance_positive",
    "performance_negative",
    "data_properties",
    "training_positive",
    "training_negative",
    "training_neutral",
    "contour_glyph",
    "sampled_points",
    "error",
]

SCATTER_MARKERS = {
    "circle": "o",
    "triangle": "^",
    "square": "s",
    "diamond": "D",
    "pentagon": "p",
    "hexagon": "h",
    "octagon": "8",
}

GLYPH_FIELD_OPTIONS = [
    "phy_loss",
    "data_loss",
    "phys_eff",
    "bc_loss",
    "u_pred_grad",
    "phy_loss_grad",
    "entk",
]

LAYER_DEFAULTS = {
    1: {
        "enabled": True,
        "source": "testing",
        "vis": "zebra_map",
        "field": "u_referance",
        "colors": "main_data",
        "alpha": 1.0,
        "levels": 15,
        "point_size": 18.0,
        "glyph_size": 0.03,
        "scatter_marker": "circle",
        "scatter_fill": "filled",
        "glyph_type": "radar",
        "glyph_fields": ("phy_loss", "data_loss", "phy_loss_grad", "u_pred_grad", "entk"),
        "show_colorbar": True,
        "range_mode": "manual",
        "vmin": -1.5,
        "vmax": 1.5,
        "tick_mode": "levels",
        "tick_count": 5,
        "tick_values": "",
        "colorbar_title": "u_ref",
        "contour_width": 1.0,
    },
    2: {
        "enabled": True,
        "source": "testing",
        "vis": "contour",
        "field": "u_pred",
        "colors": "contour_glyph",
        "alpha": 1.0,
        "levels": 15,
        "point_size": 18.0,
        "glyph_size": 0.03,
        "scatter_marker": "circle",
        "scatter_fill": "filled",
        "glyph_type": "radar",
        "glyph_fields": ("phy_loss", "data_loss", "phy_loss_grad", "u_pred_grad", "entk"),
        "show_colorbar": True,
        "range_mode": "manual",
        "vmin": -1.5,
        "vmax": 1.5,
        "tick_mode": "levels",
        "tick_count": 5,
        "tick_values": "",
        "colorbar_title": "u_pred",
        "contour_width": 1.0,
    },
    3: {
        "enabled": True,
        "source": "training",
        "vis": "scatter",
        "field": "data_loss",
        "colors": "training_negative",
        "alpha": 1.0,
        "levels": 15,
        "point_size": 32.0,
        "glyph_size": 0.03,
        "scatter_marker": "circle",
        "scatter_fill": "filled",
        "glyph_type": "radar",
        "glyph_fields": ("phy_loss", "data_loss", "phy_loss_grad", "u_pred_grad", "entk"),
        "show_colorbar": True,
        "range_mode": "auto",
        "vmin": 0.0,
        "vmax": 1.0,
        "tick_mode": "auto",
        "tick_count": 5,
        "tick_values": "",
        "colorbar_title": "trn_err",
        "contour_width": 1.0,
    },
}


@lru_cache(maxsize=4)
def _load_case(case_dir: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_dir = Path(case_dir)
    training_data = load_numeric_csv(data_dir / "training.csv")
    testing_data = load_numeric_csv(data_dir / "testing" / "20000.csv")
    return training_data, testing_data


def _discover_case_options(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


@lru_cache(maxsize=32)
def _training_epoch_data(case_dir: str, epoch: int) -> dict[str, np.ndarray]:
    training_data, _ = _load_case(case_dir)
    return filter_rows(training_data, epoch=float(epoch))


@lru_cache(maxsize=64)
def _testing_grid(case_dir: str, field: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    _, testing_data = _load_case(case_dir)
    t_values, x_values, grid = reshape_regular_grid(testing_data, field)
    extent = (float(t_values.min()), float(t_values.max()), float(x_values.min()), float(x_values.max()))
    return grid, extent


@lru_cache(maxsize=128)
def _training_grid(
    case_dir: str,
    field: str,
    epoch: int,
    t_bins: int,
    x_bins: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    epoch_data = _training_epoch_data(case_dir, epoch)
    t_range, x_range, grid = grid_scatter(epoch_data, field, t_bins=t_bins, x_bins=x_bins)
    return grid, (t_range[0], t_range[1], x_range[0], x_range[1])


def _resolve_cmap(style: str):
    cmaps = build_default_colormaps()
    return cmaps[style]


def _default_color_style(source_name: str, field: str, vis_type: str) -> str:
    if vis_type == "glyph":
        return "contour_glyph"
    if vis_type == "zebra_map":
        if field in {"u_referance", "u_pred", "u_ref"}:
            return "main_data"
        if field in {"phy_loss", "data_loss"}:
            return "performance_positive"
        if field == "bc_loss":
            return "performance_negative"
        if field in {"phys_eff", "u_pred_grad", "phy_loss_grad"}:
            return "training_positive"
        if field == "entk":
            return "data_properties"
        return "main_data"
    if field in {"u_referance", "u_pred", "u_ref"}:
        return "main_data"
    if field == "error":
        return "error"
    if field in {"phy_loss", "data_loss"}:
        return "performance_positive"
    if field in {"phys_eff", "u_pred_grad", "phy_loss_grad"}:
        return "training_positive"
    if field == "bc_loss":
        return "performance_negative"
    if field == "entk":
        return "data_properties"
    return "main_data"


def _default_extent(testing_data: dict[str, np.ndarray]) -> list[float]:
    t_values = np.unique(testing_data["t"])
    x_values = np.unique(testing_data["x"])
    return [float(t_values.min()), float(t_values.max()), float(x_values.min()), float(x_values.max())]


def _set_widget_visibility(widget, visible: bool) -> None:
    widget.layout.display = "" if visible else "none"


def _build_level_values(vmin: float, vmax: float, levels: int) -> np.ndarray:
    return np.linspace(vmin, vmax, max(2, levels))


def _build_contour_colorbar_mappable(level_values: np.ndarray, cmap, vmin: float, vmax: float) -> tuple[ScalarMappable, np.ndarray]:
    if level_values.size == 0:
        raise ValueError("Contour colorbar requires at least one level.")

    samples = np.linspace(0.0, 1.0, 512)
    rgba = np.ones((samples.size, 4), dtype=float)
    rgba[:, :3] = 1.0
    rgba[:, 3] = 1.0

    if level_values.size == 1 or np.isclose(vmax, vmin):
        band = 0.012
    else:
        normalized_levels = (level_values - vmin) / (vmax - vmin)
        min_gap = float(np.min(np.diff(normalized_levels)))
        band = max(min_gap * 0.12, 0.008)

    for idx, level in enumerate(level_values):
        normalized = 0.5 if np.isclose(vmax, vmin) else float((level - vmin) / (vmax - vmin))
        mask = np.abs(samples - normalized) <= band
        rgba[mask] = cmap(idx / max(1, level_values.size - 1))

    contour_cmap = ListedColormap(rgba, name=f"{getattr(cmap, 'name', 'contour')}_levelbar")
    contour_norm = Normalize(vmin=vmin, vmax=vmax)
    return ScalarMappable(norm=contour_norm, cmap=contour_cmap), level_values


def _resolve_tick_values(
    tick_mode: str,
    tick_count: int,
    tick_values_text: str,
    vmin: float,
    vmax: float,
    levels: int,
    vis_type: str,
) -> np.ndarray | None:
    if tick_mode == "none":
        return None
    if tick_mode == "levels" or (tick_mode == "auto" and vis_type in {"contour", "zebra_map"}):
        return _build_level_values(vmin, vmax, levels)
    if tick_mode == "count":
        return np.linspace(vmin, vmax, max(2, tick_count))
    if tick_mode == "manual":
        parts = [part.strip() for part in tick_values_text.split(",") if part.strip()]
        if not parts:
            return np.linspace(vmin, vmax, max(2, tick_count))
        return np.array([float(part) for part in parts], dtype=float)
    return np.linspace(vmin, vmax, max(2, tick_count))


def _render_spatial_grid(
    case_dir: str,
    source_name: str,
    field: str,
    epoch: int,
    t_bins: int,
    x_bins: int,
) -> tuple[np.ndarray, list[float]]:
    if source_name == "testing":
        grid, extent = _testing_grid(case_dir, field)
        return grid, list(extent)

    grid, extent = _training_grid(case_dir, field, epoch, t_bins, x_bins)
    return grid, list(extent)


def _plot_spatial_layer(
    ax: plt.Axes,
    case_dir: str,
    layer_config: dict[str, object],
    training_data: dict[str, np.ndarray],
    testing_data: dict[str, np.ndarray],
    epoch: int,
    t_bins: int,
    x_bins: int,
    colorbar_slot: int,
):
    source_name = str(layer_config["source"])
    field = str(layer_config["field"])
    vis_type = str(layer_config["vis"])
    cmap = _resolve_cmap(str(layer_config["colors"]))
    alpha = float(layer_config["alpha"])
    levels = int(layer_config["levels"])
    point_size = float(layer_config["point_size"])
    show_colorbar = bool(layer_config["show_colorbar"])
    range_mode = str(layer_config["range_mode"])
    vmin = None if range_mode == "auto" else float(layer_config["vmin"])
    vmax = None if range_mode == "auto" else float(layer_config["vmax"])
    contour_width = float(layer_config["contour_width"])
    tick_mode = str(layer_config["tick_mode"])
    tick_count = int(layer_config["tick_count"])
    tick_values_text = str(layer_config["tick_values"])
    colorbar_title = str(layer_config.get("colorbar_title", "")).strip()
    display_label = colorbar_title or f"{source_name}:{field}:{vis_type}"
    scatter_marker = str(layer_config.get("scatter_marker", "circle"))
    scatter_fill = str(layer_config.get("scatter_fill", "filled"))
    glyph_type = str(layer_config.get("glyph_type", "radar"))
    glyph_fields = list(layer_config.get("glyph_fields", []))
    glyph_radius = float(layer_config.get("glyph_size", 0.03))

    if vis_type == "glyph":
        epoch_data = _training_epoch_data(case_dir, epoch)
        metrics = [metric for metric in glyph_fields if metric in GLYPH_FIELD_OPTIONS]
        if not metrics:
            metrics = ["phy_loss", "data_loss", "phy_loss_grad", "u_pred_grad", "entk"]
        palette = [cmap(value)[:3] for value in np.linspace(0.0, 1.0, len(metrics))]
        normalized = np.vstack([_normalize_metric(epoch_data[key]) for key in metrics]).T
        for idx in range(epoch_data["t"].size):
            if glyph_type == "ring":
                _draw_ring_glyph(
                    ax,
                    float(epoch_data["t"][idx]),
                    float(epoch_data["x"][idx]),
                    normalized[idx],
                    palette,
                    radius=glyph_radius,
                )
            else:
                _draw_radar_glyph(
                    ax,
                    float(epoch_data["t"][idx]),
                    float(epoch_data["x"][idx]),
                        normalized[idx],
                        palette,
                        radius=glyph_radius,
                )
        return {
            "artist": None,
            "mappable": None,
            "show_colorbar": False,
            "label": display_label or f"training:{glyph_type}:glyph",
            "slot": colorbar_slot,
            "ticks": None,
            "legend": {
                "kind": "glyph",
                "glyph_type": glyph_type,
                "metrics": metrics,
                "colors": palette,
                "title": display_label or f"{glyph_type} glyph",
            },
        }

    grid, extent = _render_spatial_grid(case_dir, source_name, field, epoch, t_bins, x_bins)
    grid_vmin = float(np.min(grid)) if vmin is None else vmin
    grid_vmax = float(np.max(grid)) if vmax is None else vmax
    norm = Normalize(vmin=grid_vmin, vmax=grid_vmax)

    if vis_type == "heatmap":
        artist = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, alpha=alpha, aspect="equal", norm=norm)
        tick_values = _resolve_tick_values(
            tick_mode, tick_count, tick_values_text, grid_vmin, grid_vmax, levels, vis_type
        )
        return {
            "artist": artist,
            "mappable": ScalarMappable(norm=norm, cmap=cmap),
            "show_colorbar": show_colorbar,
            "label": display_label,
            "slot": colorbar_slot,
            "ticks": tick_values,
        }

    if vis_type == "zebra_map":
        level_values = _build_level_values(grid_vmin, grid_vmax, levels)
        zebra_cmap = make_zebra_cmap(
            cmap,
            grid_vmin,
            grid_vmax,
            levels=levels,
            name=f"{getattr(cmap, 'name', 'custom')}_dynamic_zebra_{levels}",
        )
        artist = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap=zebra_cmap,
            alpha=alpha,
            aspect="equal",
            norm=norm,
        )
        tick_values = _resolve_tick_values(
            tick_mode, tick_count, tick_values_text, grid_vmin, grid_vmax, levels, vis_type
        )
        return {
            "artist": artist,
            "mappable": ScalarMappable(norm=norm, cmap=zebra_cmap),
            "show_colorbar": show_colorbar,
            "label": display_label,
            "slot": colorbar_slot,
            "ticks": tick_values,
            "level_values": level_values,
        }

    if vis_type == "contour":
        level_values = _build_level_values(grid_vmin, grid_vmax, levels)
        if source_name == "testing":
            t_axis = np.linspace(extent[0], extent[1], grid.shape[1])
            x_axis = np.linspace(extent[2], extent[3], grid.shape[0])
            mesh_t, mesh_x = np.meshgrid(t_axis, x_axis)
            contour_source = ("grid", mesh_t, mesh_x, grid)
        else:
            source = _training_epoch_data(case_dir, epoch)
            triangulation = mtri.Triangulation(source["t"], source["x"])
            contour_source = ("tri", triangulation, source[field])

        artist = ScalarMappable(norm=norm, cmap=cmap)
        if contour_source[0] == "grid":
            _, mesh_t, mesh_x, grid_values = contour_source
            ax.contour(
                mesh_t,
                mesh_x,
                grid_values,
                levels=level_values,
                cmap=cmap,
                alpha=min(1.0, alpha + 0.05),
                linewidths=contour_width,
                norm=norm,
            )
        else:
            _, triangulation, values = contour_source
            ax.tricontour(
                triangulation,
                values,
                levels=level_values,
                cmap=cmap,
                alpha=min(1.0, alpha + 0.05),
                linewidths=contour_width,
                norm=norm,
            )
        tick_values = _resolve_tick_values(
            tick_mode, tick_count, tick_values_text, grid_vmin, grid_vmax, levels, vis_type
        )
        return {
            "artist": artist,
            "mappable": artist,
            "show_colorbar": show_colorbar,
            "label": display_label,
            "slot": colorbar_slot,
            "ticks": tick_values,
            "level_values": level_values,
            "colorbar_kind": "contour_levels",
        }

    if vis_type == "scatter":
        if source_name == "testing":
            source = testing_data
        else:
            source = _training_epoch_data(case_dir, epoch)
        values = source[field]
        if range_mode == "auto":
            norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(values))
        marker_code = SCATTER_MARKERS.get(scatter_marker, "o")
        if scatter_fill == "hollow":
            edge_colors = colors.copy()
            edge_colors[:, -1] = alpha
            artist = ax.scatter(
                source["t"],
                source["x"],
                s=point_size,
                marker=marker_code,
                facecolors="none",
                edgecolors=edge_colors,
                linewidths=0.8,
            )
        else:
            colors[:, -1] = alpha
            artist = ax.scatter(source["t"], source["x"], c=colors, s=point_size, marker=marker_code, linewidths=0.0)
        tick_values = _resolve_tick_values(
            tick_mode, tick_count, tick_values_text, float(norm.vmin), float(norm.vmax), levels, vis_type
        )
        return {
            "artist": artist,
            "mappable": ScalarMappable(norm=norm, cmap=cmap),
            "show_colorbar": show_colorbar,
            "label": display_label,
            "slot": colorbar_slot,
            "ticks": tick_values,
            "legend": None,
            "colorbar_kind": "continuous",
        }

    raise ValueError(f"Unsupported spatial visualization: {vis_type}")


def render_multiplexing_view(
    case_dir: str | Path,
    *,
    layers: list[dict[str, object]],
    epoch: int,
    show_colorbar: bool,
    t_bins: int,
    x_bins: int,
) -> plt.Figure:
    case_dir = str(Path(case_dir).resolve())
    training_data, testing_data = _load_case(case_dir)
    active_spatial = [layer for layer in layers if layer["enabled"] and layer["source"] in {"testing", "training"}]
    fig, ax_main = plt.subplots(figsize=(7.65, 7.65), constrained_layout=True)

    colorbar_items: list[dict[str, object]] = []
    legend_items: list[dict[str, object]] = []
    for slot, layer in enumerate(active_spatial):
        layer_result = _plot_spatial_layer(
            ax_main,
            case_dir,
            layer,
            training_data,
            testing_data,
            epoch,
            t_bins,
            x_bins,
            slot,
        )
        colorbar_items.append(layer_result)
        if layer_result.get("legend") is not None:
            legend_items.append(layer_result["legend"])

    extent = _default_extent(testing_data)
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_xlabel("t")
    ax_main.set_ylabel("x")
    ax_main.set_aspect("equal", adjustable="box")

    visible_colorbars = []
    if show_colorbar:
        visible_colorbars = [item for item in colorbar_items if item["show_colorbar"] and item["mappable"] is not None]

    if visible_colorbars or legend_items:
        fig.canvas.draw()
        main_box = ax_main.get_position()
        if legend_items and len(visible_colorbars) >= 2:
            target_right = 0.56
        elif legend_items:
            target_right = 0.61
        elif len(visible_colorbars) >= 3:
            target_right = 0.68
        else:
            target_right = 0.72
        new_width = max(0.42, target_right - main_box.x0)
        ax_main.set_position([main_box.x0, main_box.y0, new_width, main_box.height])
        fig.canvas.draw()
        main_box = ax_main.get_position()

        if visible_colorbars:
            bar_width = 0.013
            bar_gap = 0.065
            bar_height = main_box.height * 0.9
            bar_bottom = main_box.y0 + (main_box.height - bar_height) / 2.0
            bar_left = main_box.x1 + 0.065

            for idx, item in enumerate(visible_colorbars):
                cax = fig.add_axes(
                    [
                        bar_left + idx * (bar_width + bar_gap),
                        bar_bottom,
                        bar_width,
                        bar_height,
                    ]
                )
                mappable = item["mappable"]
                ticks = item.get("ticks")
                if item.get("colorbar_kind") == "contour_levels":
                    mappable, ticks = _build_contour_colorbar_mappable(
                        np.asarray(item["level_values"], dtype=float),
                        mappable.cmap,
                        float(mappable.norm.vmin),
                        float(mappable.norm.vmax),
                    )
                colorbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
                colorbar.ax.set_title(textwrap.fill(str(item["label"]), width=7), fontsize=7, pad=14)
                colorbar.ax.tick_params(labelsize=7)

        if legend_items:
            legend_left = bar_left + len(visible_colorbars) * (bar_width + bar_gap) + 0.045
            legend_width = 0.16
            legend_height = min(0.28, (main_box.height - 0.02) / max(1, len(legend_items)))
            legend_gap = 0.03
            for idx, legend in enumerate(legend_items):
                top = main_box.y1 - idx * (legend_height + legend_gap)
                bottom = top - legend_height
                lax = fig.add_axes([legend_left, max(main_box.y0, bottom), legend_width, legend_height])
                draw_glyph_legend(
                    lax,
                    str(legend["glyph_type"]),
                    list(legend["metrics"]),
                    list(legend["colors"]),
                )
                lax.text(0.02, 1.02, textwrap.fill(str(legend["title"]), width=16), transform=lax.transAxes, ha="left", va="bottom", fontsize=8)

    return fig


def create_interactive_multiplexing_ui(repo_root: str | Path):
    import ipywidgets as widgets
    from IPython.display import display

    repo_root = Path(repo_root)
    data_root = repo_root / "data"
    case_options = _discover_case_options(data_root)
    default_case = "wave_case" if "wave_case" in case_options else (case_options[0] if case_options else "")

    data_source = widgets.Dropdown(options=case_options, value=default_case, description="Dataset")
    epoch = widgets.IntSlider(value=20000, min=0, max=20000, step=2000, description="Epoch", continuous_update=False)
    t_bins = widgets.IntSlider(value=250, min=50, max=700, step=50, description="t bins", continuous_update=False)
    x_bins = widgets.IntSlider(value=250, min=50, max=700, step=50, description="x bins", continuous_update=False)
    export_name = widgets.Text(value="multiplexing-view.png", description="Filename", continuous_update=False)
    save_button = widgets.Button(description="Save figure", button_style="success")
    save_status = widgets.HTML("")

    layer_specs: list[dict[str, object]] = []

    def _make_layer_box(index: int):
        defaults = LAYER_DEFAULTS[index]
        enabled = widgets.Checkbox(value=defaults["enabled"], description=f"Enable L{index}")
        source = widgets.Dropdown(
            options=["testing", "training"],
            value=defaults["source"],
            description=f"Source {index}",
        )
        vis = widgets.Dropdown(options=VIS_OPTIONS[source.value], value=defaults["vis"], description=f"Vis {index}")
        field_options = SPATIAL_FIELDS[source.value]
        field = widgets.Dropdown(options=field_options, value=defaults["field"], description=f"Field {index}")
        colors = widgets.Dropdown(
            options=COLOR_STYLE_OPTIONS,
            value=defaults["colors"],
            description=f"Colors {index}",
        )
        alpha = widgets.FloatSlider(value=defaults["alpha"], min=0.1, max=1.0, step=0.05, description=f"Alpha {index}", continuous_update=False)
        levels = widgets.IntSlider(value=defaults["levels"], min=4, max=40, step=1, description=f"Levels {index}", continuous_update=False)
        point_size = widgets.FloatSlider(value=defaults["point_size"], min=2.0, max=120.0, step=2.0, description=f"Point {index}", continuous_update=False)
        glyph_size = widgets.FloatSlider(value=defaults["glyph_size"], min=0.01, max=0.08, step=0.005, description=f"G Size {index}", continuous_update=False)
        scatter_marker = widgets.Dropdown(options=list(SCATTER_MARKERS.keys()), value=defaults["scatter_marker"], description=f"Marker {index}")
        scatter_fill = widgets.Dropdown(options=["filled", "hollow"], value=defaults["scatter_fill"], description=f"Fill {index}")
        glyph_type = widgets.Dropdown(options=["radar", "ring"], value=defaults["glyph_type"], description=f"Glyph {index}")
        glyph_fields = widgets.SelectMultiple(
            options=GLYPH_FIELD_OPTIONS,
            value=defaults["glyph_fields"],
            description=f"Metrics {index}",
            rows=6,
        )
        show_layer_colorbar = widgets.Checkbox(value=defaults["show_colorbar"], description=f"ColorBar {index}")
        range_mode = widgets.Dropdown(options=["auto", "manual"], value=defaults["range_mode"], description=f"Range {index}")
        vmin = widgets.FloatText(value=defaults["vmin"], description=f"vmin {index}")
        vmax = widgets.FloatText(value=defaults["vmax"], description=f"vmax {index}")
        tick_mode = widgets.Dropdown(options=["auto", "levels", "count", "manual", "none"], value=defaults["tick_mode"], description=f"Ticks {index}")
        tick_count = widgets.IntSlider(value=defaults["tick_count"], min=2, max=15, step=1, description=f"Tick n {index}", continuous_update=False)
        tick_values = widgets.Text(value=defaults["tick_values"], description=f"Tick vals {index}", continuous_update=False)
        colorbar_title = widgets.Text(value=defaults["colorbar_title"], description=f"CB Title {index}", continuous_update=False)
        contour_width = widgets.FloatSlider(value=defaults["contour_width"], min=0.2, max=3.0, step=0.2, description=f"Line {index}", continuous_update=False)
        settings_box = widgets.VBox(
            [
                source,
                vis,
                field,
                colors,
                alpha,
                levels,
                point_size,
                glyph_size,
                scatter_marker,
                scatter_fill,
                glyph_type,
                glyph_fields,
                show_layer_colorbar,
                range_mode,
                vmin,
                vmax,
                tick_mode,
                tick_count,
                tick_values,
                colorbar_title,
                contour_width,
            ]
        )

        def _sync_options(*_args):
            vis_options = VIS_OPTIONS[source.value]
            vis.options = vis_options
            if vis.value not in vis_options:
                vis.value = vis_options[0]

            new_fields = SPATIAL_FIELDS[source.value]
            field.options = new_fields
            if field.value not in new_fields:
                field.value = new_fields[0]

            recommended_color = _default_color_style(source.value, field.value, vis.value)
            if colors.value not in COLOR_STYLE_OPTIONS or colors.value == "":
                colors.value = recommended_color
            elif vis.value == "glyph":
                colors.value = "contour_glyph"

            uses_levels = vis.value in {"contour", "zebra_map"}
            uses_points = vis.value == "scatter"
            uses_line_width = vis.value == "contour"
            uses_glyph = vis.value == "glyph"
            uses_scatter_controls = vis.value == "scatter"
            uses_colorbar = vis.value not in {"line", "glyph"}
            uses_manual_range = range_mode.value == "manual"
            uses_tick_count = tick_mode.value == "count"
            uses_tick_values = tick_mode.value == "manual"
            uses_colorbar_title = uses_colorbar and show_layer_colorbar.value

            _set_widget_visibility(field, not uses_glyph)
            _set_widget_visibility(levels, uses_levels)
            _set_widget_visibility(point_size, uses_points)
            _set_widget_visibility(glyph_size, uses_glyph)
            _set_widget_visibility(scatter_marker, uses_scatter_controls)
            _set_widget_visibility(scatter_fill, uses_scatter_controls)
            _set_widget_visibility(glyph_type, uses_glyph)
            _set_widget_visibility(glyph_fields, uses_glyph)
            _set_widget_visibility(show_layer_colorbar, uses_colorbar)
            _set_widget_visibility(range_mode, uses_colorbar)
            _set_widget_visibility(vmin, uses_colorbar and uses_manual_range)
            _set_widget_visibility(vmax, uses_colorbar and uses_manual_range)
            _set_widget_visibility(tick_mode, uses_colorbar)
            _set_widget_visibility(tick_count, uses_colorbar and uses_tick_count)
            _set_widget_visibility(tick_values, uses_colorbar and uses_tick_values)
            _set_widget_visibility(colorbar_title, uses_colorbar_title)
            _set_widget_visibility(contour_width, uses_line_width)
            _set_widget_visibility(settings_box, enabled.value)

            if vis.value in {"contour", "zebra_map"} and tick_mode.value == "auto":
                tick_mode.value = "levels"
            if vis.value not in {"contour", "zebra_map"} and tick_mode.value == "levels":
                tick_mode.value = "auto"

        enabled.observe(_sync_options, names="value")
        source.observe(_sync_options, names="value")
        vis.observe(_sync_options, names="value")
        range_mode.observe(_sync_options, names="value")
        tick_mode.observe(_sync_options, names="value")
        show_layer_colorbar.observe(_sync_options, names="value")
        _sync_options()

        layer_specs.append(
            {
                "enabled": enabled,
                "source": source,
                "vis": vis,
                "field": field,
                "colors": colors,
                "alpha": alpha,
                "levels": levels,
                "point_size": point_size,
                "glyph_size": glyph_size,
                "scatter_marker": scatter_marker,
                "scatter_fill": scatter_fill,
                "glyph_type": glyph_type,
                "glyph_fields": glyph_fields,
                "show_colorbar": show_layer_colorbar,
                "range_mode": range_mode,
                "vmin": vmin,
                "vmax": vmax,
                "tick_mode": tick_mode,
                "tick_count": tick_count,
                "tick_values": tick_values,
                "colorbar_title": colorbar_title,
                "contour_width": contour_width,
            }
        )

        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.HTML(f"<b>Layer {index}</b>"),
                        enabled,
                    ],
                    layout=widgets.Layout(justify_content="space-between", align_items="center", width="100%"),
                ),
                settings_box,
            ],
            layout=widgets.Layout(width="32%"),
        )

    layer_boxes = [_make_layer_box(1), _make_layer_box(2), _make_layer_box(3)]
    output = widgets.Output()

    def _collect_layers() -> list[dict[str, object]]:
        layers: list[dict[str, object]] = []
        for spec in layer_specs:
            layers.append(
                {
                    "enabled": spec["enabled"].value,
                    "source": spec["source"].value,
                    "vis": spec["vis"].value,
                    "field": spec["field"].value,
                    "colors": spec["colors"].value,
                    "alpha": spec["alpha"].value,
                    "levels": spec["levels"].value,
                    "point_size": spec["point_size"].value,
                    "glyph_size": spec["glyph_size"].value,
                    "scatter_marker": spec["scatter_marker"].value,
                    "scatter_fill": spec["scatter_fill"].value,
                    "glyph_type": spec["glyph_type"].value,
                    "glyph_fields": list(spec["glyph_fields"].value),
                    "show_colorbar": spec["show_colorbar"].value,
                    "range_mode": spec["range_mode"].value,
                    "vmin": spec["vmin"].value,
                    "vmax": spec["vmax"].value,
                    "tick_mode": spec["tick_mode"].value,
                    "tick_count": spec["tick_count"].value,
                    "tick_values": spec["tick_values"].value,
                    "colorbar_title": spec["colorbar_title"].value,
                    "contour_width": spec["contour_width"].value,
                }
            )
        return layers

    def _build_figure() -> plt.Figure:
        return render_multiplexing_view(
            data_root / data_source.value,
            layers=_collect_layers(),
            epoch=epoch.value,
            show_colorbar=True,
            t_bins=t_bins.value,
            x_bins=x_bins.value,
        )

    def _render(*_args):
        with output:
            output.clear_output(wait=True)
            fig = _build_figure()
            display(fig)
            plt.close(fig)

    def _save_current(_button):
        figures_dir = repo_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        filename = export_name.value.strip() or "multiplexing-view.png"
        if not filename.lower().endswith(".png"):
            filename = f"{filename}.png"
        output_path = figures_dir / filename
        fig = _build_figure()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        save_status.value = f"Saved to {output_path}"

    save_button.on_click(_save_current)

    controls = [data_source, epoch, t_bins, x_bins]
    for spec in layer_specs:
        controls.extend(spec.values())
    for control in controls:
        control.observe(_render, names="value")

    _render()

    global_controls = widgets.VBox(
        [
            widgets.HTML("<b>Global Controls</b>"),
            widgets.HBox(
                [
                    widgets.VBox([data_source, epoch], layout=widgets.Layout(width="48%")),
                    widgets.VBox([t_bins, x_bins], layout=widgets.Layout(width="48%")),
                    widgets.VBox([export_name, save_button, save_status], layout=widgets.Layout(width="48%")),
                ],
                layout=widgets.Layout(justify_content="space-between", align_items="flex-start", width="100%"),
            ),
        ]
    )
    layers_panel = widgets.VBox(
        [
            widgets.HTML("<b>Layer Controls</b>"),
            widgets.HBox(
                layer_boxes,
                layout=widgets.Layout(justify_content="space-between", align_items="flex-start", width="100%"),
            ),
        ]
    )
    ui = widgets.VBox(
        [
            widgets.HTML("<h3>Interactive Multiplexing Explorer</h3>"),
            widgets.HTML("<p>This notebook accompanies the paper's supplementary material and is intended for reviewers and readers to interactively test the multiplexing design. Choose a dataset, configure up to three layers, and compare how data source, visualization, field selection, and color design affect the final view.</p>"),
            global_controls,
            layers_panel,
            output,
        ]
    )
    return ui
