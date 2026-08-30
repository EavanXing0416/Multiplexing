# Multiplexing

Interactive supplementary-material repository for the multiplexing visualization design used in the paper.

## Online Demo

Reviewers and readers can open the interactive demo directly in a browser, without installing Python or Jupyter locally, by launching the notebook through Binder and Voilà:

[Launch Interactive Demo on Binder](https://mybinder.org/v2/gh/EavanXing0416/Multiplexing/main?urlpath=voila%2Frender%2Fnotebooks%2Finteractive_multiplexing.ipynb)

Notes:

- The first launch may take a few minutes while Binder builds the environment.
- After the environment is ready, the notebook will open as a clean interactive app rendered by Voilà.
- No local installation is required for normal review and testing.

## What This Demo Provides

This repository accompanies the paper's supplementary material and is intended to help reviewers and readers test the multiplexing workflow on the released example dataset.

The interactive demo supports:

- selecting the released dataset
- choosing the training epoch
- enabling up to three visualization layers
- comparing `heatmap`, `zebra_map`, `contour`, `scatter`, and multivariate `glyph` views
- adjusting colormap range, ticks, colorbars, and per-layer styling
- exporting the current figure as a PNG

## Repository Layout

- `data/wave_case/`: minimal CSV files needed to reproduce the example
- `notebooks/interactive_multiplexing.ipynb`: interactive notebook used for the online demo
- `src/multiplexing/`: reusable loading, color, and plotting utilities
- `scripts/generate_figures.py`: one-command figure generation entrypoint

## Local Use

If a reader prefers to run the notebook locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open:

`notebooks/interactive_multiplexing.ipynb`

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
