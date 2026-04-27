
"""
3D Cell Segmentation Pipeline
==============================
Segments cells in Leica .lif confocal files and computes 3D morphological metrics.

Usage:
    python pipeline.py --lif_file path/to/file.lif --output_dir results/

Requirements:
    pip install readlif cellpose scikit-image matplotlib numpy scipy pandas
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from readlif.reader import LifFile
from cellpose import models
from skimage.measure import regionprops, marching_cubes, mesh_surface_area
from skimage.exposure import rescale_intensity


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def norm(img, p_low=1, p_high=99.5):
    lo, hi = np.percentile(img, p_low), np.percentile(img, p_high)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)


def get_best_z(img_obj, channel=1, tile=0):
    """Returns the Z slice index with the highest mean intensity."""
    z_means = [
        np.array(img_obj.get_frame(z=z, t=0, c=channel, m=tile)).mean()
        for z in range(img_obj.dims.z)
    ]
    return int(np.argmax(z_means))


def load_stack(img_obj, z_start, z_end, channel, tile):
    """Loads a Z stack for a given channel and tile."""
    return np.stack([
        np.array(img_obj.get_frame(z=z, t=0, c=channel, m=tile))
        for z in range(z_start, z_end)
    ])


def compute_3d_mask(stack_ch1, cell_mask_2d):
    """
    Extends a 2D cell mask through Z using the FWHM criterion.
    Returns (cell_3d, active_z) or (None, None) if too few slices.
    """
    z_profile = np.array([
        stack_ch1[z][cell_mask_2d].mean()
        for z in range(stack_ch1.shape[0])
    ])
    active_z = np.where(z_profile >= z_profile.max() * 0.5)[0]

    if len(active_z) < 3:
        return None, None

    cell_3d = np.zeros_like(stack_ch1, dtype=bool)
    for z in active_z:
        cell_3d[z] = cell_mask_2d

    return cell_3d, active_z


def compute_metrics(cell_3d, active_z, region, stack_ch1,
                    stack_ch0, voxel_xy, voxel_z, voxel_vol, ch0_thr):
    """Computes all 3D metrics for a single cell."""
    volume_um3 = cell_3d.sum() * voxel_vol

    try:
        verts, faces, _, _ = marching_cubes(
            cell_3d.astype(np.uint8), level=0.5,
            spacing=(voxel_z, voxel_xy, voxel_xy)
        )
        surface_um2 = mesh_surface_area(verts, faces)
        sphericity  = min(
            (np.pi**(1/3) * (6 * volume_um3)**(2/3)) / surface_um2, 1.0
        )
    except Exception:
        surface_um2 = np.nan
        sphericity  = np.nan

    feret_max    = region.axis_major_length * voxel_xy
    feret_min    = region.axis_minor_length * voxel_xy
    feret_ratio  = feret_min / (feret_max + 1e-8)

    ch0_vals      = stack_ch0[cell_3d]
    ch0_mean      = float(ch0_vals.mean())
    ch0_total     = float(ch0_vals.sum())
    ch0_vol_um3   = float((ch0_vals > ch0_thr).sum()) * voxel_vol
    ch0_vol_ratio = ch0_vol_um3 / (volume_um3 + 1e-8)

    return {
        'volume_um3':     round(volume_um3, 2),
        'surface_um2':    round(surface_um2, 2) if not np.isnan(surface_um2) else np.nan,
        'sphericity':     round(sphericity, 4)  if not np.isnan(sphericity)  else np.nan,
        'feret_max_um':   round(feret_max, 2),
        'feret_min_um':   round(feret_min, 2),
        'feret_ratio':    round(feret_ratio, 4),
        'ch1_mean':       round(float(stack_ch1[cell_3d].mean()), 2),
        'ch0_mean':       round(ch0_mean, 2),
        'ch0_total':      round(ch0_total, 2),
        'ch0_volume_um3': round(ch0_vol_um3, 2),
        'ch0_vol_ratio':  round(ch0_vol_ratio, 4),
        'cell_height_um': round(len(active_z) * voxel_z, 2),
        'n_active_z':     len(active_z),
    }


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_results(df, output_dir):
    metrics = {
        'volume_um3':     'Volume (µm³)',
        'surface_um2':    'Surface Area (µm²)',
        'sphericity':     'Sphericity',
        'feret_max_um':   'Feret Max (µm)',
        'feret_ratio':    'Feret Ratio (min/max)',
        'ch0_mean':       'Ch0 Mean Intensity',
        'ch0_vol_ratio':  'Ch0 / Cell Volume Ratio',
        'cell_height_um': 'Cell Height (µm)',
    }

    conditions = df.condition.unique().tolist()
    palette    = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63',
                  '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A']
    colors     = {c: palette[i] for i, c in enumerate(conditions)}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10),
                             facecolor='#111111')
    fig.suptitle('3D Cell Morphology Comparison', fontsize=15, y=1.01,
                 color='white')
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics.items()):
        data = [df[df.condition==c][col].dropna().values for c in conditions]
        bp   = ax.boxplot(
            data, patch_artist=True, widths=0.5,
            medianprops=dict(color='white', linewidth=2),
            flierprops=dict(marker='.', markersize=3, alpha=0.3)
        )
        for patch, cond in zip(bp['boxes'], conditions):
            patch.set_facecolor(colors[cond])
            patch.set_alpha(0.8)
        for i, (cond, vals) in enumerate(zip(conditions, data)):
            ax.scatter(np.random.normal(i+1, 0.07, len(vals)),
                       vals, alpha=0.3, s=8, color=colors[cond], zorder=5)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_title(label, fontsize=11, pad=6)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'cell_morphology_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(lif_file, output_dir, z_margin=10, gpu=True):
    """
    Full pipeline: load → segment → measure → save.

    Parameters
    ----------
    lif_file   : path to .lif file
    output_dir : folder to save results
    z_margin   : Z slices above/below best Z to include (default 10)
    gpu        : use GPU for Cellpose (default True)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'cell_results_all_tiles.csv')

    if os.path.exists(save_path):
        os.remove(save_path)

    # Load file
    print(f"\nLoading: {lif_file}")
    lif        = LifFile(lif_file)
    all_series = list(lif.get_iter_image())
    print(f"Found {len(all_series)} conditions: "
          f"{[s.name for s in all_series]}")

    # Load Cellpose
    print("\nLoading Cellpose model...")
    model = models.CellposeModel(gpu=gpu)
    print("✅ Cellpose ready")

    # Process each condition
    for img_obj in all_series:
        condition = img_obj.name
        scale     = img_obj.scale
        voxel_xy  = 1 / scale[0]
        voxel_z   = 1 / scale[2]
        voxel_vol = voxel_xy * voxel_xy * voxel_z
        n_z       = img_obj.dims.z
        n_tiles   = img_obj.dims.m
        bit_depth = img_obj.bit_depth[0]
        ch0_thr   = 0.04 * (2**bit_depth - 1)

        print(f"\n{'='*55}")
        print(f"Condition : {condition}  |  Tiles: {n_tiles}  |  Z: {n_z}")
        print(f"Voxel XY  : {voxel_xy:.4f} µm  |  Z: {voxel_z:.4f} µm")
        print(f"{'='*55}")

        # Auto-detect best Z
        best_z  = get_best_z(img_obj)
        z_start = max(0, best_z - z_margin)
        z_end   = min(n_z, best_z + z_margin)
        print(f"Best Z: {best_z}  |  Window: Z={z_start}–{z_end}")

        for tile_idx in range(n_tiles):
            print(f"\n  Tile {tile_idx+1:>2}/{n_tiles} ...", end=' ', flush=True)

            # Load stacks
            stack_ch1 = load_stack(img_obj, z_start, z_end, channel=1, tile=tile_idx)
            stack_ch0 = load_stack(img_obj, z_start, z_end, channel=0, tile=tile_idx)

            # Normalize for Cellpose
            best_z_local = int(np.argmax([s.mean() for s in stack_ch1]))
            frame_norm   = rescale_intensity(
                stack_ch1[best_z_local].astype(np.float32), out_range=(0, 255)
            ).astype(np.uint8)

            # 2D segmentation
            masks_2d, _, _ = model.eval(
                frame_norm, diameter=None,
                flow_threshold=0.4, cellprob_threshold=0.0
            )
            print(f"2D: {masks_2d.max()} cells", end=' ', flush=True)

            rows = []
            for region in regionprops(masks_2d):
                cell_mask_2d = masks_2d == region.label

                # Exclude border cells
                bb = region.bbox
                if bb[0]==0 or bb[1]==0: continue
                if bb[2]==masks_2d.shape[0] or bb[3]==masks_2d.shape[1]: continue

                # Build 3D mask via FWHM
                cell_3d, active_z = compute_3d_mask(stack_ch1, cell_mask_2d)
                if cell_3d is None:
                    continue

                # Compute metrics
                m = compute_metrics(
                    cell_3d, active_z, region,
                    stack_ch1, stack_ch0,
                    voxel_xy, voxel_z, voxel_vol, ch0_thr
                )
                m.update({
                    'condition': condition,
                    'tile':      tile_idx,
                    'cell_id':   region.label,
                })
                rows.append(m)

            # Save tile immediately
            if rows:
                df_tile      = pd.DataFrame(rows)
                write_header = not os.path.exists(save_path)
                df_tile.to_csv(save_path, mode='a',
                               header=write_header, index=False)
                print(f"→ {len(rows)} saved")
            else:
                print("→ 0 kept")

    # Final summary
    df = pd.read_csv(save_path)
    print(f"\n{'='*55}")
    print(f"DONE — Total cells: {len(df)}")
    print(f"{'='*55}")
    print(df.groupby('condition')[
        ['volume_um3', 'sphericity', 'feret_ratio', 'ch0_vol_ratio']
    ].mean().round(3))

    # Save summary CSV
    metrics_cols = ['volume_um3', 'surface_um2', 'sphericity',
                    'feret_max_um', 'feret_min_um', 'feret_ratio',
                    'ch0_mean', 'ch0_vol_ratio', 'cell_height_um']
    summary = df.groupby('condition')[metrics_cols].agg(['mean', 'std']).round(3)
    summary.columns = [f'{c}_{s}' for c, s in summary.columns]
    summary.to_csv(os.path.join(output_dir, 'cell_morphology_summary.csv'))

    # Plot
    plot_results(df, output_dir)

    print(f"\nFiles saved to: {output_dir}")
    print(f"  cell_results_all_tiles.csv")
    print(f"  cell_morphology_summary.csv")
    print(f"  cell_morphology_results.png")

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D cell segmentation pipeline for Leica .lif files"
    )
    parser.add_argument('--lif_file',   required=True,
                        help='Path to .lif file')
    parser.add_argument('--output_dir', default='results/',
                        help='Output directory (default: results/)')
    parser.add_argument('--z_margin',   type=int, default=10,
                        help='Z slices each side of best Z (default: 10)')
    parser.add_argument('--no_gpu',     action='store_true',
                        help='Disable GPU (default: GPU enabled)')
    args = parser.parse_args()

    run_pipeline(
        lif_file   = args.lif_file,
        output_dir = args.output_dir,
        z_margin   = args.z_margin,
        gpu        = not args.no_gpu,
    )