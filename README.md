# Multiplexing

This repository provides the interactive supplementary-material demo for the multiplexing visualization design presented in the paper. It is intended for readers who would like to inspect the released example data, test different visualization combinations, and export figures directly from the browser.

## Open The Demo

The recommended public entry point is the lightweight Panel app:

[Launch Interactive Demo](https://mybinder.org/v2/gh/EavanXing0416/Multiplexing/main?urlpath=%2Fpanel%2Fapp)

Notes:

- The demo runs in the browser through Binder and does not require local installation.
- The first launch may take a few minutes while Binder builds the environment.
- The repository also includes a notebook interface for local exploration.

## What Readers Can Test

The interactive demo allows users to:

- choose the released example dataset
- multiplex up to three visualization layers in a single figure
- combine `heatmap`, `zebra_map`, `contour line`, `scatter (univariate glyph)`, and `multivariate glyph` views
- adjust colormaps, transparency, contour levels, scatter styles, and colorbar settings
- explore different training epochs for training-based layers
- export the current visualization as a PNG

## How To Use The Demo

1. Open the Binder link above.
2. Wait for the environment to finish loading.
3. In `Global Controls`, choose the dataset, training epoch, grid resolution, and export filename.
4. In `Layer Controls`, enable up to three layers and configure each layer independently.
5. Use `Source` to switch between `testing` and `training` data.
6. Use `Vis` to choose the rendering style for a layer.
7. Adjust `Field`, `Colors`, `Range`, `Ticks`, and `ColorBar` settings as needed.
8. Use `Download PNG` to save the current figure.

## Important Data Note

`testing` layers use the released testing snapshot stored in:

`data/wave_case/testing/20000.csv`

`training` layers are drawn from:

`data/wave_case/training.csv`

This file contains multiple saved training epochs, so changing the `Epoch` control affects training-based layers but not the fixed testing snapshot.

## Included Files

- `app.py`: lightweight standalone Panel app used for the recommended public demo
- `notebooks/interactive_multiplexing.ipynb`: notebook-based interface
- `data/wave_case/`: released example data required by the demo
- `src/multiplexing/`: plotting, color, and data-loading utilities
- `scripts/generate_figures.py`: batch figure-generation entrypoint

## Run Locally

To run the Panel app locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
panel serve app.py --show
```

To run the notebook interface locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open:

`notebooks/interactive_multiplexing.ipynb`

## Alternative Notebook Demo

If a notebook-based Binder session is preferred, the repository also provides:

[Launch Notebook Demo](https://mybinder.org/v2/gh/EavanXing0416/Multiplexing/main?urlpath=voila%2Frender%2Fnotebooks%2Finteractive_multiplexing.ipynb)

