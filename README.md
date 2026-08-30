# Multiplexing

Interactive supplementary-material repository for the multiplexing visualization design used in the paper.

## Recommended Interface

The recommended interactive interface is the lightweight Panel app:

`app.py`

Compared with the notebook-based version, the Panel app:

- avoids most notebook widget overhead
- uses an explicit `Update Figure` action for more stable interaction
- keeps the same multiplexing configuration workflow
- supports figure export as PNG

## Online Access

Reviewers and readers can open the lightweight Panel app directly in a browser, without installing Python or Jupyter locally, through Binder:

[Launch Panel App on Binder](https://mybinder.org/v2/gh/EavanXing0416/Multiplexing/main?urlpath=%2Fpanel%2Fapp)

Notes:

- The first launch may take a few minutes while Binder builds the environment.
- After the environment is ready, Binder will open the Panel app served from `app.py`.
- No local installation is required for normal review and testing.
- The repository still includes the notebook interface as a secondary option if needed.

## What This Demo Provides

This repository accompanies the paper's supplementary material and is intended to help reviewers and readers test the multiplexing workflow on the released example dataset.

The interactive demo supports:

- selecting the released dataset
- choosing the training epoch
- enabling up to three visualization layers
- comparing `heatmap`, `zebra_map`, `contour`, `scatter`, and multivariate `glyph` views
- adjusting colormap range, ticks, colorbars, and per-layer styling
- exporting the current figure as a PNG

## Local Launch

The lightweight Panel app can be launched locally with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
panel serve app.py --show
```

If a reader prefers the notebook interface instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open:

`notebooks/interactive_multiplexing.ipynb`

For a notebook-based Binder session, the previous Voilà entry remains available:

[Launch Notebook Demo on Binder](https://mybinder.org/v2/gh/EavanXing0416/Multiplexing/main?urlpath=voila%2Frender%2Fnotebooks%2Finteractive_multiplexing.ipynb)

## Repository Layout

- `app.py`: lightweight standalone Panel app
- `notebooks/interactive_multiplexing.ipynb`: notebook-based interactive demo
- `data/wave_case/`: minimal CSV files needed to reproduce the example
- `src/multiplexing/`: reusable loading, color, and plotting utilities
- `scripts/generate_figures.py`: one-command figure generation entrypoint

## Included Data

This repository keeps only the files needed to reproduce the multiplexing example:

- `training.csv`
- `training_loss.csv`
- `testing/20000.csv`

These files were copied from:

`VirtualVis/v-vis/public/data/pde/wave/2025-11-28_13-10-20-420730/`

## Provenance

The original exploratory notebook was:

`/Users/yiwenxing/Documents/Documents - Yiwen’s MacBook Pro/VirtualVis/parsingWFC1s0a.ipynb`
