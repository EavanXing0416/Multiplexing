from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .colors import build_default_colormaps
from .io import filter_rows, load_numeric_csv


def reshape_regular_grid(data: dict[str, np.ndarray], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_values = np.unique(data["t"])
    x_values = np.unique(data["x"])
    grid = data[value_key].reshape(x_values.size, t_values.size)
    return t_values, x_values, grid


def grid_scatter(
    data: dict[str, np.ndarray],
    value_key: str,
    *,
    t_bins: int = 300,
    x_bins: int = 300,
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray]:
    tmin, tmax = float(np.min(data["t"])), float(np.max(data["t"]))
    xmin, xmax = float(np.min(data["x"])), float(np.max(data["x"]))
    t_edges = np.linspace(tmin, tmax, t_bins + 1)
    x_edges = np.linspace(xmin, xmax, x_bins + 1)

    t_idx = np.clip(np.searchsorted(t_edges, data["t"], side="right") - 1, 0, t_bins - 1)
    x_idx = np.clip(np.searchsorted(x_edges, data["x"], side="right") - 1, 0, x_bins - 1)

    acc = np.zeros((x_bins, t_bins), dtype=float)
    cnt = np.zeros((x_bins, t_bins), dtype=float)
    np.add.at(acc, (x_idx, t_idx), data[value_key])
    np.add.at(cnt, (x_idx, t_idx), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = acc / cnt
    grid[np.isnan(grid)] = 0.0
    return (tmin, tmax), (xmin, xmax), grid


def plot_training_loss(training_loss: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_loss["epoch"], training_loss["total_loss"], color="#1f4e79", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_prediction_error(
    testing_data: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    cmaps = build_default_colormaps()
    t_values, x_values, u_ref = reshape_regular_grid(testing_data, "u_referance")
    _, _, u_pred = reshape_regular_grid(testing_data, "u_pred")
    _, _, u_err = reshape_regular_grid(testing_data, "error")
    extent = [t_values.min(), t_values.max(), x_values.min(), x_values.max()]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        (u_ref, cmaps["data_zebra"], r"$u_{ref}(t, x)$"),
        (u_pred, cmaps["data_zebra"], r"$u_{\theta}(t, x)$"),
        (u_err, cmaps["error"], r"$|u_{ref} - u_{\theta}|$"),
    ]

    for ax, (grid, cmap, title) in zip(axes, panels, strict=True):
        im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(values, [2, 98])
    clipped = np.clip(values, low, high)
    denom = high - low
    if np.isclose(denom, 0.0):
        return np.zeros_like(clipped)
    return (clipped - low) / denom


def _draw_radar_glyph(
    ax: plt.Axes,
    t: float,
    x: float,
    normalized_values: np.ndarray,
    colors: list[str],
    *,
    radius: float = 0.03,
) -> None:
    count = normalized_values.size
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    vertices = []
    for angle, value in zip(angles, normalized_values, strict=True):
        scale = radius * (0.35 + 0.65 * value)
        vertices.append((t + scale * np.cos(angle), x + scale * np.sin(angle)))
    polygon = Polygon(vertices, closed=True, facecolor="none", edgecolor="black", linewidth=0.35, alpha=0.9)
    ax.add_patch(polygon)

    for angle, value, color in zip(angles, normalized_values, colors, strict=True):
        scale = radius * (0.35 + 0.65 * value)
        triangle = Polygon(
            [(t, x), (t + scale * np.cos(angle - 0.22), x + scale * np.sin(angle - 0.22)), (t + scale * np.cos(angle + 0.22), x + scale * np.sin(angle + 0.22))],
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=0.78,
        )
        ax.add_patch(triangle)


def plot_local_multiplexing(
    training_data: dict[str, np.ndarray],
    testing_data: dict[str, np.ndarray],
    output_path: Path,
    *,
    epoch: int = 20000,
) -> None:
    cmaps = build_default_colormaps()
    train_epoch = filter_rows(training_data, epoch=float(epoch))
    t_range, x_range, error_grid = grid_scatter(testing_data, "error", t_bins=500, x_bins=500)
    extent = [t_range[0], t_range[1], x_range[0], x_range[1]]

    metrics = ["phy_loss", "data_loss", "phy_loss_grad", "u_pred_grad", "entk"]
    palette = ["#476911", "#144f80", "#6d199e", "#9e1943", "#9e4119"]
    normalized = np.vstack([_normalize_metric(train_epoch[key]) for key in metrics]).T

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(error_grid, origin="lower", extent=extent, cmap=cmaps["error"], aspect="equal", alpha=0.92)
    for idx in range(train_epoch["t"].size):
        _draw_radar_glyph(ax, float(train_epoch["t"][idx]), float(train_epoch["x"][idx]), normalized[idx], palette)

    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(f"Local Multiplexing at Epoch {epoch}")
    fig.colorbar(im, ax=ax, shrink=0.72, label="Prediction error")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=metric)
        for metric, color in zip(metrics, palette, strict=True)
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(data_dir: str | Path, output_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = load_numeric_csv(data_dir / "training.csv")
    training_loss = load_numeric_csv(data_dir / "training_loss.csv")
    testing_data = load_numeric_csv(data_dir / "testing" / "20000.csv")

    plot_training_loss(training_loss, output_dir / "training-loss.png")
    plot_reference_prediction_error(testing_data, output_dir / "reference-prediction-error.png")
    plot_local_multiplexing(training_data, testing_data, output_dir / "local-multiplexing.png")
