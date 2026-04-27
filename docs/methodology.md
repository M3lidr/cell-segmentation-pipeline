with open('docs/methodology.md', 'w') as f:
    f.write("""# Methodology

## Data

- **Format**: Leica `.lif` confocal microscopy file
- **Microscope**: Leica DMI8-CS, resonant confocal scanner
- **Objective**: 40× water immersion, NA 1.1
- **Conditions**: WT (Wild Type), N1, N2, N3
- **Tiles**: 20 tiles for WT, 25 tiles each for N1/N2/N3
- **Channels**: Alexa 405 (Ch0 — organelles), Alexa 488 (Ch1 — cell membrane)
- **Z-slices**: 118–131 per tile
- **Voxel size**: extracted dynamically from metadata (~0.29 × 0.29 × 0.77 µm)

---

## Step 1 — File Loading

Library: `readlif`. All metadata extracted programmatically — voxel size, tile count,
Z count, bit depth. No physical parameters are hardcoded anywhere in the pipeline.

Voxel size derived from `img_obj.scale` (pixels per µm), inverted to µm per pixel.
Noise threshold derived from bit depth: `0.04 × (2^bit_depth - 1)`.

---

## Step 2 — Optimal Z Selection

Mean intensity of Ch1 computed across all Z slices using tile 0 as representative.
Z slice with maximum mean intensity selected as the focal plane.
A 20-slice window (±10 slices) extracted around this Z — covers ~15 µm depth,
sufficient to fully capture each cell in 3D.

Validated: FWHM result is identical for ±10 and ±20 slice windows (Min/Peak = 0.03),
confirming signal fully drops to background within the narrow window.

---

## Step 3 — 2D Cell Segmentation

Tool: **Cellpose v4**, model `cyto3` (whole-cell segmentation).

Cellpose works by predicting, for every pixel, the direction toward the nearest cell
center (optical flow). Pixels whose flows converge to the same point are grouped as
one cell. This allows accurate separation of touching cells without manual tuning.

Applied on the single brightest Z slice per tile. Automatic diameter estimation used.

Parameters:
- `flow_threshold = 0.4` — default, controls strictness of flow grouping
- `cellprob_threshold = 0.0` — include all candidate cells
- `diameter = None` — auto-estimated per tile

---

## Step 4 — 3D Volume Reconstruction (2.5D approach)

Full 3D Cellpose was evaluated but too slow for the dataset size (>20 min/tile on GPU).
Instead a 2.5D approach is used:

1. For each 2D cell mask, compute mean Ch1 intensity at every Z slice within the window
2. Apply **FWHM criterion**: keep only Z slices where signal ≥ 50% of the cell's own peak
3. Build 3D mask by extending the 2D footprint through those active Z slices

The FWHM (Full Width at Half Maximum) criterion is the standard in microscopy for
defining the physical extent of a signal. It is per-cell and fully dynamic.

---

## Step 5 — Cell Filtering

Three filters applied:

1. **Border cells**: any cell whose bounding box touches the image edge is excluded.
   Rationale: partial cells have artificially small/incorrect measurements.

2. **Minimum Z extent**: cells spanning fewer than 3 Z slices excluded.
   Rationale: likely debris or out-of-focus noise, not real cells.

3. **FWHM minimum**: at least 3 active Z slices required after FWHM filtering.

Typical retention rate: ~65–75% of Cellpose detections pass all filters.

---

## Step 6 — 3D Metric Computation

| Metric | Formula / Method |
|---|---|
| Volume (µm³) | voxel_count × (voxel_xy² × voxel_z) |
| Surface area (µm²) | Marching cubes (scikit-image), spacing=(voxel_z, voxel_xy, voxel_xy) |
| Sphericity | π^(1/3) × (6V)^(2/3) / A — ranges 0 to 1 |
| Feret max (µm) | region.axis_major_length × voxel_xy |
| Feret min (µm) | region.axis_minor_length × voxel_xy |
| Feret ratio | feret_min / feret_max |
| Ch0 mean intensity | mean(Ch0 signal inside 3D cell mask) |
| Ch0 total intensity | sum(Ch0 signal inside 3D cell mask) |
| Ch0 volume (µm³) | count(Ch0 voxels > noise_threshold) × voxel_vol |
| Ch0 volume ratio | Ch0_volume / cell_volume |
| Cell height (µm) | n_active_z_slices × voxel_z |

---

## Step 7 — Validation

| Check | Method | Result |
|---|---|---|
| Visual | Outlines overlaid on raw images, random tiles | ✅ Accurate |
| Sanity | Compared to theoretical 15µm perfect sphere | ✅ Consistent |
| Tile consistency | Per-tile mean volume bar chart | ✅ No outlier tiles |
| Z stability | ±10 vs ±20 window FWHM comparison | ✅ Identical result |
| Outlier sensitivity | IQR filtering before/after comparison | ✅ <5% change |

---

## Dependencies

- `readlif` — Leica .lif file parsing
- `cellpose` — deep learning cell segmentation
- `scikit-image` — marching cubes, region properties
- `scipy` — signal processing
- `pandas` — data management
- `matplotlib` — visualization
""")
print("✅ docs/methodology.md created")

