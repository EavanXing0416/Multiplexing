from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .colors import build_default_colormaps
from .io import filter_rows, load_numeric_csv
from .plots import _draw_radar_glyph, _normalize_metric, grid_scatter, reshape_regular_grid


@lru_cache(maxsize=4)
def _load_case(case_dir: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_dir = Path(case_dir)
    training_data = load_numeric_csv(data_dir / "training.csv")
    testing_data = load_numeric_csv(data_dir / "testing" / "20000.csv")
    return training_data, testing_data


def _resolve_cmap(style: str):
    cmaps = build_default_colormaps()
    return {
        "zebra-data": cmaps["data_zebra"],
        "zebra-phy": cmaps["phy_zebra"],
        "greys": cmaps["error"],
        "viridis": plt.get_cmap("viridis"),
        "coolwarm": plt.get_cmap("coolwarm"),
        "magma": plt.get_cmap("magma"),
    }[style]


def render_multiplexing_view(
    case_dir: str | Path,
    *,
    layer_type: str,
    field: str,
    color_style: str,
    alpha: float,
    levels: int,
    point_size: float,
    glyph_radius: float,
    epoch: int,
    show_colorbar: bool,
    t_bins: int,
    x_bins: int,
) -> plt.Figure:
    training_data, testing_data = _load_case(str(Path(case_dir).resolve()))
    cmap = _resolve_cmap(color_style)
    fig, ax = plt.subplots(figsize=(8, 8))

    if field in {"u_referance", "u_pred", "error"}:
        t_values, x_values, grid = reshape_regular_grid(testing_data, field)
        extent = [t_values.min(), t_values.max(), x_values.min(), x_values.max()]
    else:
        epoch_data = filter_rows(training_data, epoch=float(epoch))
        t_range, x_range, grid = grid_scatter(epoch_data, field, t_bins=t_bins, x_bins=x_bins)
        extent = [t_range[0], t_range[1], x_range[0], x_range[1]]

    image = None
    if layer_type == "heatmap":
        image = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, alpha=alpha, aspect="equal")
    elif layer_type == "contour":
        t_axis = np.linspace(extent[0], extent[1], grid.shape[1])
        x_axis = np.linspace(extent[2], extent[3], grid.shape[0])
        mesh_t, mesh_x = np.meshgrid(t_axis, x_axis)
        image = ax.contourf(mesh_t, mesh_x, grid, levels=levels, cmap=cmap, alpha=alpha)
    elif layer_type == "scatter":
        source = testing_data if field in {"u_referance", "u_pred", "error"} else filter_rows(training_data, epoch=float(epoch))
        values = source[field]
        norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
        colors = cmap(norm(values))
        colors[:, -1] = alpha
        image = ax.scatter(source["t"], source["x"], c=colors, s=point_size, linewidths=0.0)
    elif layer_type == "glyph":
        error_epoch = filter_rows(training_data, epoch=float(epoch))
        _, _, error_grid = grid_scatter(testing_data, "error", t_bins=t_bins, x_bins=x_bins)
        image = ax.imshow(error_grid, origin="lower", extent=extent, cmap=build_default_colormaps()["error"], alpha=0.9, aspect="equal")
        metrics = ["phy_loss", "data_loss", "phy_loss_grad", "u_pred_grad", "entk"]
        palette = ["#476911", "#144f80", "#6d199e", "#9e1943", "#9e4119"]
        normalized = np.vstack([_normalize_metric(error_epoch[key]) for key in metrics]).T
        for idx in range(error_epoch["t"].size):
            _draw_radar_glyph(
                ax,
                float(error_epoch["t"][idx]),
                float(error_epoch["x"][idx]),
                normalized[idx],
                palette,
                radius=glyph_radius,
            )
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

    ax.set_title(f"{layer_type.title()} view for {field}")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    if show_colorbar and image is not None and layer_type != "scatter":
        fig.colorbar(image, ax=ax, shrink=0.76, label=field)

    fig.tight_layout()
    return fig


def create_interactive_multiplexing_ui(repo_root: str | Path):
    import ipywidgets as widgets
    from IPython.display import display

    repo_root = Path(repo_root)
    case_dir = repo_root / "data" / "wave_case"

    layer_type = widgets.Dropdown(
        options=["heatmap", "contour", "scatter", "glyph"],
        value="heatmap",
        description="Layer",
    )
    field = widgets.Dropdown(
        options=[
            "u_referance",
            "u_pred",
            "error",
            "phy_loss",
            "data_loss",
            "u_pred_grad",
            "entk",
        ],
        value="error",
        description="Field",
    )
    color_style = widgets.Dropdown(
        options=["greys", "zebra-data", "zebra-phy", "viridis", "coolwarm", "magma"],
        value="greys",
        description="Colors",
    )
    alpha = widgets.FloatSlider(value=0.9, min=0.1, max=1.0, step=0.05, description="Alpha")
    levels = widgets.IntSlider(value=15, min=4, max=40, step=1, description="Levels")
    point_size = widgets.FloatSlider(value=18.0, min=2.0, max=120.0, step=2.0, description="Point size")
    glyph_radius = widgets.FloatSlider(value=0.03, min=0.01, max=0.08, step=0.005, description="Glyph size")
    epoch = widgets.IntSlider(value=20000, min=0, max=20000, step=2000, description="Epoch")
    show_colorbar = widgets.Checkbox(value=True, description="Show colorbar")
    t_bins = widgets.IntSlider(value=300, min=50, max=700, step=50, description="t bins")
    x_bins = widgets.IntSlider(value=300, min=50, max=700, step=50, description="x bins")

    output = widgets.Output()

    def _render(*_args):
        with output:
            output.clear_output(wait=True)
            fig = render_multiplexing_view(
                case_dir,
                layer_type=layer_type.value,
                field=field.value,
                color_style=color_style.value,
                alpha=alpha.value,
                levels=levels.value,
                point_size=point_size.value,
                glyph_radius=glyph_radius.value,
                epoch=epoch.value,
                show_colorbar=show_colorbar.value,
                t_bins=t_bins.value,
                x_bins=x_bins.value,
            )
            display(fig)
            plt.close(fig)

    controls = [
        layer_type,
        field,
        color_style,
        alpha,
        levels,
        point_size,
        glyph_radius,
        epoch,
        show_colorbar,
        t_bins,
        x_bins,
    ]
    for control in controls:
        control.observe(_render, names="value")

    _render()

    left = widgets.VBox([layer_type, field, color_style, alpha, levels, show_colorbar])
    right = widgets.VBox([point_size, glyph_radius, epoch, t_bins, x_bins])
    ui = widgets.VBox(
        [
            widgets.HTML("<h3>Interactive Multiplexing Explorer</h3>"),
            widgets.HBox([left, right]),
            output,
        ]
    )
    return ui
