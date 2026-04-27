# cell-segmentation-pipeline
with open('README.md', 'w') as f:
    f.write("""# Cell Segmentation Pipeline

Automated 3D morphological analysis of confocal microscopy data from Leica `.lif` files using Cellpose.

## What it does

- Loads a Leica `.lif` file and extracts all metadata automatically (voxel size, tile count, Z slices, bit depth)
- Detects the optimal focal Z plane per condition
- Segments cells in 2D using Cellpose (`cyto3` model)
- Reconstructs 3D cell volumes using the FWHM criterion
- Computes 3D metrics for every cell across all tiles and conditions
- Saves per-cell measurements and summary statistics to CSV

## Metrics computed

| Metric | Description |
|---|---|
| Volume (µm³) | Voxel count × voxel volume |
| Surface area (µm²) | Marching cubes algorithm |
| Sphericity | π^(1/3) × (6V)^(2/3) / A — 1.0 = perfect sphere |
| Feret max/min (µm) | Longest and shortest cell axis |
| Feret ratio | min/max — 1.0 = perfectly round |
| Ch0 mean intensity | Mean organelle signal inside cell |
| Ch0 volume ratio | Fraction of cell occupied by organelle signal |
| Cell height (µm) | Z extent via FWHM |

## Data format

Designed for Leica `.lif` confocal files with:
- Multiple conditions as image series
- Multiple mosaic tiles per condition
- 2 fluorescence channels (Ch1 = cell membrane, Ch0 = organelle)
- 3D Z-stack per tile

## Usage

### In Google Colab (recommended)
Open `notebooks/cell_segmentation.ipynb`, mount your Google Drive, set `LIF_FILE` to your file path, and run all cells.

### As a script
```bash
pip install -r requirements.txt
python src/pipeline.py --lif_file path/to/your_file.lif --output_dir results/
```

## Results

- 2580 cells measured across 4 conditions (WT, N1, N2, N3)
- 20–25 mosaic tiles per condition fully processed
- N1 cells ~2× larger in volume vs WT (3846 vs 2054 µm³)
- All conditions show consistent sphericity (~0.93–0.97)

## Validation

- Visual spot check across random tiles confirmed accurate segmentation
- Measurements validated against theoretical perfect sphere values
- Tile-to-tile consistency confirmed — no outlier tiles
- Z window stability confirmed — FWHM result independent of window size
- Outlier sensitivity confirmed — IQR filtering changes means by <5%

## Methods

See [docs/methodology.md](docs/methodology.md) for full step-by-step methodology.

## Requirements

- Python 3.10+
- GPU recommended (NVIDIA T4 or better) for Cellpose
- Google Colab works out of the box with free T4 GPU

## Author

Muhammad Ali Aldribi  
BSc Computer Science, NYU Abu Dhabi  
Microscopy Lab, NYUAD
""")
print("✅ README.md created")