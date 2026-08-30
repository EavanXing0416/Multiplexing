# Multiplexing

Clean supplementary-material version of the multiplexing figure pipeline from the earlier `VirtualVis` project.

## Repository layout

- `data/wave_case/`: minimal CSV files needed to reproduce the paper figures.
- `src/multiplexing/`: reusable loading, color, and plotting utilities.
- `scripts/generate_figures.py`: one-command figure generation entrypoint.
- `figures/`: generated outputs.

## Included data

This repository keeps only the files needed to reproduce the multiplexing example:

- `training.csv`
- `training_loss.csv`
- `testing/20000.csv`

These files were copied from:

`VirtualVis/v-vis/public/data/pde/wave/2025-11-28_13-10-20-420730/`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_figures.py
```

Generated figures will be written to `figures/`.

## Interactive notebook

An interactive notebook is included at `notebooks/interactive_multiplexing.ipynb`.

It exposes UI controls for:

- layer type
- data field
- colorbar / colormap style
- alpha
- contour levels
- point size
- glyph radius
- epoch

Start Jupyter from the repository root and open the notebook:

```bash
jupyter notebook
```

## What was cleaned up

- Removed the dependency on the old `v-vis` frontend project.
- Replaced hard-coded notebook paths with a small, portable project structure.
- Kept only the core plotting logic needed for supplementary-material figures.
- Dropped exploratory notebook cells, repeated helper functions, and embedded notebook output.

## Current outputs

The script produces:

1. `training-loss.png`
2. `reference-prediction-error.png`
3. `local-multiplexing.png`

## Provenance

The original exploratory notebook was:

`/Users/yiwenxing/Documents/Documents - Yiwen’s MacBook Pro/VirtualVis/parsingWFC1s0a.ipynb`
