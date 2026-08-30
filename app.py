from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import panel as pn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from multiplexing.interactive import (  # noqa: E402
    COLOR_STYLE_OPTIONS,
    GLYPH_FIELD_OPTIONS,
    LAYER_DEFAULTS,
    SCATTER_MARKERS,
    SPATIAL_FIELDS,
    VIS_OPTIONS,
    _default_color_style,
    _discover_case_options,
    render_multiplexing_view,
)

pn.extension(sizing_mode="stretch_width")


class LayerControls:
    def __init__(self, index: int):
        self.index = index
        defaults = LAYER_DEFAULTS[index]
        self.enabled = pn.widgets.Checkbox(name=f"Enable L{index}", value=defaults["enabled"])
        self.source = pn.widgets.Select(name="Source", options=["testing", "training"], value=defaults["source"])
        self.vis = pn.widgets.Select(name="Vis", options=VIS_OPTIONS[self.source.value], value=defaults["vis"])
        self.field = pn.widgets.Select(name="Field", options=SPATIAL_FIELDS[self.source.value], value=defaults["field"])
        self.colors = pn.widgets.Select(
            name="Colors",
            options=COLOR_STYLE_OPTIONS,
            value=defaults["colors"],
        )
        self.alpha = pn.widgets.FloatSlider(name="Alpha", start=0.1, end=1.0, step=0.05, value=defaults["alpha"])
        self.levels = pn.widgets.IntSlider(name="Levels", start=4, end=40, step=1, value=defaults["levels"])
        self.point_size = pn.widgets.FloatSlider(name="Point", start=2.0, end=120.0, step=2.0, value=defaults["point_size"])
        self.glyph_size = pn.widgets.FloatSlider(name="G Size", start=0.01, end=0.08, step=0.005, value=defaults["glyph_size"])
        self.scatter_marker = pn.widgets.Select(name="Marker", options=list(SCATTER_MARKERS.keys()), value=defaults["scatter_marker"])
        self.scatter_fill = pn.widgets.Select(name="Fill", options=["filled", "hollow"], value=defaults["scatter_fill"])
        self.glyph_type = pn.widgets.Select(name="Glyph", options=["radar", "ring"], value=defaults["glyph_type"])
        self.glyph_fields = pn.widgets.MultiSelect(
            name="Metrics",
            options=GLYPH_FIELD_OPTIONS,
            value=list(defaults["glyph_fields"]),
            size=7,
        )
        self.show_colorbar = pn.widgets.Checkbox(name="ColorBar", value=defaults["show_colorbar"])
        self.range_mode = pn.widgets.Select(name="Range", options=["auto", "manual"], value=defaults["range_mode"])
        self.vmin = pn.widgets.FloatInput(name="vmin", value=defaults["vmin"], step=0.1)
        self.vmax = pn.widgets.FloatInput(name="vmax", value=defaults["vmax"], step=0.1)
        self.tick_mode = pn.widgets.Select(name="Ticks", options=["auto", "levels", "count", "manual", "none"], value=defaults["tick_mode"])
        self.tick_count = pn.widgets.IntSlider(name="Tick n", start=2, end=15, step=1, value=defaults["tick_count"])
        self.tick_values = pn.widgets.TextInput(name="Tick vals", value=defaults["tick_values"])
        self.colorbar_title = pn.widgets.TextInput(name="CB Title", value=defaults["colorbar_title"])
        self.contour_width = pn.widgets.FloatSlider(name="Line", start=0.2, end=3.0, step=0.2, value=defaults["contour_width"])

        self._controls = [
            self.source,
            self.vis,
            self.field,
            self.colors,
            self.alpha,
            self.levels,
            self.point_size,
            self.glyph_size,
            self.scatter_marker,
            self.scatter_fill,
            self.glyph_type,
            self.glyph_fields,
            self.show_colorbar,
            self.range_mode,
            self.vmin,
            self.vmax,
            self.tick_mode,
            self.tick_count,
            self.tick_values,
            self.colorbar_title,
            self.contour_width,
        ]

        self.source.param.watch(self._sync, "value")
        self.vis.param.watch(self._sync, "value")
        self.range_mode.param.watch(self._sync, "value")
        self.tick_mode.param.watch(self._sync, "value")
        self.show_colorbar.param.watch(self._sync, "value")
        self.enabled.param.watch(self._sync, "value")
        self._sync()

    def _sync(self, *_events):
        vis_options = VIS_OPTIONS[self.source.value]
        self.vis.options = vis_options
        if self.vis.value not in vis_options:
            self.vis.value = vis_options[0]

        field_options = SPATIAL_FIELDS[self.source.value]
        self.field.options = field_options
        if self.field.value not in field_options:
            self.field.value = field_options[0]

        recommended = _default_color_style(self.source.value, self.field.value, self.vis.value)
        if self.vis.value == "glyph":
            self.colors.value = "contour_glyph"
        elif self.colors.value not in COLOR_STYLE_OPTIONS:
            self.colors.value = recommended

        uses_levels = self.vis.value in {"contour", "zebra_map"}
        uses_points = self.vis.value == "scatter"
        uses_glyph = self.vis.value == "glyph"
        uses_scatter_controls = self.vis.value == "scatter"
        uses_line_width = self.vis.value == "contour"
        uses_colorbar = self.vis.value != "glyph"
        uses_manual_range = self.range_mode.value == "manual"
        uses_tick_count = self.tick_mode.value == "count"
        uses_tick_values = self.tick_mode.value == "manual"

        self.field.visible = not uses_glyph
        self.levels.visible = uses_levels
        self.point_size.visible = uses_points
        self.glyph_size.visible = uses_glyph
        self.scatter_marker.visible = uses_scatter_controls
        self.scatter_fill.visible = uses_scatter_controls
        self.glyph_type.visible = uses_glyph
        self.glyph_fields.visible = uses_glyph
        self.show_colorbar.visible = uses_colorbar
        self.range_mode.visible = uses_colorbar
        self.vmin.visible = uses_colorbar and uses_manual_range
        self.vmax.visible = uses_colorbar and uses_manual_range
        self.tick_mode.visible = uses_colorbar
        self.tick_count.visible = uses_colorbar and uses_tick_count
        self.tick_values.visible = uses_colorbar and uses_tick_values
        self.colorbar_title.visible = uses_colorbar and self.show_colorbar.value
        self.contour_width.visible = uses_line_width
        for control in self._controls:
            control.disabled = not self.enabled.value

        if self.vis.value in {"contour", "zebra_map"} and self.tick_mode.value == "auto":
            self.tick_mode.value = "levels"
        if self.vis.value not in {"contour", "zebra_map"} and self.tick_mode.value == "levels":
            self.tick_mode.value = "auto"

    def to_config(self) -> dict[str, object]:
        return {
            "enabled": self.enabled.value,
            "source": self.source.value,
            "vis": self.vis.value,
            "field": self.field.value,
            "colors": self.colors.value,
            "alpha": self.alpha.value,
            "levels": self.levels.value,
            "point_size": self.point_size.value,
            "glyph_size": self.glyph_size.value,
            "scatter_marker": self.scatter_marker.value,
            "scatter_fill": self.scatter_fill.value,
            "glyph_type": self.glyph_type.value,
            "glyph_fields": list(self.glyph_fields.value),
            "show_colorbar": self.show_colorbar.value,
            "range_mode": self.range_mode.value,
            "vmin": self.vmin.value,
            "vmax": self.vmax.value,
            "tick_mode": self.tick_mode.value,
            "tick_count": self.tick_count.value,
            "tick_values": self.tick_values.value,
            "colorbar_title": self.colorbar_title.value,
            "contour_width": self.contour_width.value,
        }

    def panel(self) -> pn.Column:
        header = pn.Row(
            pn.pane.Markdown(f"**Layer {self.index}**", margin=(0, 0, 0, 0)),
            self.enabled,
            align="center",
        )
        return pn.Column(
            header,
            *self._controls,
            sizing_mode="stretch_width",
            styles={"border": "1px solid #d9d9d9", "border-radius": "8px", "padding": "12px", "background": "#fafafa"},
        )


case_options = _discover_case_options(ROOT / "data")
default_case = "wave_case" if "wave_case" in case_options else (case_options[0] if case_options else "")

dataset = pn.widgets.Select(name="Dataset", options=case_options, value=default_case)
epoch = pn.widgets.IntSlider(name="Epoch", start=0, end=20000, step=2000, value=20000)
t_bins = pn.widgets.IntSlider(name="t bins", start=50, end=700, step=50, value=250)
x_bins = pn.widgets.IntSlider(name="x bins", start=50, end=700, step=50, value=250)
filename = pn.widgets.TextInput(name="Filename", value="multiplexing-view.png")

layers = [LayerControls(1), LayerControls(2), LayerControls(3)]
figure_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)


def build_figure():
    fig = render_multiplexing_view(
        ROOT / "data" / dataset.value,
        layers=[layer.to_config() for layer in layers],
        epoch=epoch.value,
        show_colorbar=True,
        t_bins=t_bins.value,
        x_bins=x_bins.value,
    )
    return fig


def update_figure(_event=None):
    fig = build_figure()
    figure_pane.object = fig


def _download_callback():
    fig = build_figure()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer


download = pn.widgets.FileDownload(
    callback=_download_callback,
    filename=filename.value,
    label="Download PNG",
    button_type="success",
)


def _sync_download_name(_event):
    name = filename.value.strip() or "multiplexing-view.png"
    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    download.filename = name


filename.param.watch(_sync_download_name, "value")

for widget in [dataset, epoch, t_bins, x_bins, filename]:
    widget.param.watch(update_figure, "value")

for layer in layers:
    for control in layer._controls + [layer.enabled]:
        control.param.watch(update_figure, "value")

intro = pn.pane.Markdown(
    """
# Interactive Multiplexing Explorer

This app accompanies the paper's supplementary material and is intended to help reviewers and readers test the multiplexing design directly in the browser. Configure up to three layers, compare visualization strategies, and export the current figure for inspection.
""",
    sizing_mode="stretch_width",
)

global_controls = pn.Column(
    "## Global Controls",
    pn.Row(
        pn.Column(dataset, epoch, sizing_mode="stretch_width"),
        pn.Column(t_bins, x_bins, sizing_mode="stretch_width"),
        pn.Column(filename, download, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    ),
)

layer_controls = pn.Column(
    "## Layer Controls",
    pn.Row(*(layer.panel() for layer in layers), sizing_mode="stretch_width"),
)

app = pn.Column(
    intro,
    global_controls,
    layer_controls,
    figure_pane,
    sizing_mode="stretch_width",
    max_width=1600,
)

update_figure()

app.servable(title="Interactive Multiplexing Explorer")
