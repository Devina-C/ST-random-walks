#!/usr/bin/env python3
"""
proseg_region_viz.py
====================
4-panel visualisation of a region of interest from the Proseg pilot:
  Panel 1: DAPI only
  Panel 2: DAPI + Cellpose boundaries (before)
  Panel 3: DAPI + Proseg boundaries (after)
  Panel 4: DAPI + Proseg boundaries + transcripts (tiny dots)

Default ROI: top-right 4 tiles of pilot (r0c3, r0c4, r1c3, r1c4)
  DS x[8700:12816] y[0:6300]

Edit ROI_*_DS constants to inspect any region.

Run:
  sbatch --job-name=region_viz --partition=cpu --mem=16G
         --cpus-per-task=2 --time=00:30:00
         --output=logs/region_viz_%j.out
         --wrap="cd /scratch/users/k22026807/masters/project/resegmentation &&
                 conda run -n xenium python -u scripts/proseg_region_viz.py"
"""

import os, gc, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import spatialdata as sd
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import box as shapely_box
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

XENIUM_ZARR     = "/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr"
MERGED_NPY      = "data/merged/merged_masks.npy"
TRANSCRIPTS_PAR = ("/scratch/users/k22026807/masters/project/xenium_output/"
                   "BC_prime.zarr/points/transcripts/points.parquet")
PROSEG_PARQUET  = "data/proseg_pilot/merged_proseg_pilot.parquet"
OUT_DIR         = "data/proseg_pilot/viz"
os.makedirs(OUT_DIR, exist_ok=True)

DS_UM = 0.425

# ── Region of interest ────────────────────────────────────────────────────────
# Top-right 4 tiles: r0c3, r0c4, r1c3, r1c4
# Includes halos so tile boundaries are visible inside the image
ROI_X0_DS = 8700
ROI_X1_DS = 12816
ROI_Y0_DS = 0
ROI_Y1_DS = 6300

ROI_X0_UM = ROI_X0_DS * DS_UM
ROI_X1_UM = ROI_X1_DS * DS_UM
ROI_Y0_UM = ROI_Y0_DS * DS_UM
ROI_Y1_UM = ROI_Y1_DS * DS_UM

# Tile core boundary positions (DS coords)
CORE_X_DS = [3000, 6000, 9000, 12000]
CORE_Y_DS = [3000, 6000, 9000]

# Transcript dots — very small so boundaries remain readable
MAX_TX_PLOT         = 100000
TX_DOT_SIZE         = 0.15
TX_ASSIGNED_ALPHA   = 0.55
TX_UNASSIGNED_ALPHA = 0.30


def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}", flush=True)


def load_dapi_roi():
    ts("Loading DAPI...")
    sdata  = sd.read_zarr(XENIUM_ZARR)
    img_dt = sdata.images[list(sdata.images.keys())[0]]
    for lv in range(1, 5):
        try:
            ds_  = img_dt[f'scale{lv}'].ds
            var  = list(ds_.data_vars)[0]
            arr  = ds_[var]
            img  = (arr[0].values if arr.ndim == 3 else arr.values).astype(np.float32)
            del sdata, arr, ds_
            print(f"  Level {lv}: {img.shape}", flush=True)
            break
        except Exception:
            continue
    crop    = img[ROI_Y0_DS:ROI_Y1_DS, ROI_X0_DS:ROI_X1_DS]
    nonzero = crop[crop > 0]
    if len(nonzero):
        p1, p99 = np.percentile(nonzero, [1, 99])
        crop    = np.clip((crop - p1) / (p99 - p1 + 1e-6), 0, 1)
    ts(f"DAPI crop: {crop.shape}")
    del img; gc.collect()
    return crop


def draw_tile_lines(ax):
    for xb in CORE_X_DS:
        xb_um = xb * DS_UM
        if ROI_X0_UM <= xb_um <= ROI_X1_UM:
            ax.axvline(xb_um, color='red', lw=0.6, ls='--', alpha=0.75, zorder=5)
    for yb in CORE_Y_DS:
        yb_um = yb * DS_UM
        if ROI_Y0_UM <= yb_um <= ROI_Y1_UM:
            ax.axhline(yb_um, color='red', lw=0.6, ls='--', alpha=0.75, zorder=5)


def style_ax(ax, title):
    ax.set_facecolor('black')
    ax.set_xlim(ROI_X0_UM, ROI_X1_UM)
    ax.set_ylim(ROI_Y1_UM, ROI_Y0_UM)
    ax.set_title(title, color='white', fontsize=9, pad=6)
    ax.tick_params(colors='white', labelsize=6)
    ax.set_xlabel('x (µm)', color='white', fontsize=7)
    ax.set_ylabel('y (µm)', color='white', fontsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')


def draw_proseg_boundaries(ax, gdf, color='cyan', lw=0.35, alpha=0.85):
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            if poly.is_empty: continue
            xs_p, ys_p = poly.exterior.xy
            ax.plot(xs_p, ys_p, color=color, lw=lw, alpha=alpha)


if __name__ == "__main__":
    t0 = time.time()
    ts("=== PROSEG REGION VIZ ===")
    ts(f"ROI DS  x[{ROI_X0_DS}:{ROI_X1_DS}] y[{ROI_Y0_DS}:{ROI_Y1_DS}]")
    ts(f"ROI µm  x[{ROI_X0_UM:.0f}:{ROI_X1_UM:.0f}] y[{ROI_Y0_UM:.0f}:{ROI_Y1_UM:.0f}]")

    extent = [ROI_X0_UM, ROI_X1_UM, ROI_Y1_UM, ROI_Y0_UM]
    roi_w  = ROI_X1_DS - ROI_X0_DS
    roi_h  = ROI_Y1_DS - ROI_Y0_DS
    aspect = roi_h / roi_w

    # ── Load data ──────────────────────────────────────────────────────────────
    dapi = load_dapi_roi()

    ts("Loading Cellpose masks...")
    merged_full = np.load(MERGED_NPY, mmap_mode='r')
    merged_roi  = merged_full[ROI_Y0_DS:ROI_Y1_DS,
                               ROI_X0_DS:ROI_X1_DS].copy()
    del merged_full; gc.collect()
    n_cp = np.unique(merged_roi[merged_roi > 0]).shape[0]
    ts(f"Cellpose cells: {n_cp:,}")

    ts("Loading Proseg shapes...")
    gdf_all = gpd.read_parquet(PROSEG_PARQUET)
    roi_box = shapely_box(ROI_X0_UM, ROI_Y0_UM, ROI_X1_UM, ROI_Y1_UM)
    gdf_roi = gdf_all[gdf_all.intersects(roi_box)].copy()
    del gdf_all; gc.collect()
    ts(f"Proseg cells: {len(gdf_roi):,}")

    ts("Loading transcripts...")
    pts_roi = pd.read_parquet(
    TRANSCRIPTS_PAR,
    columns=['x', 'y', 'feature_name', 'cell_id'],
    filters=[
        ('x', '>=', ROI_X0_UM),
        ('x', '<',  ROI_X1_UM),
        ('y', '>=', ROI_Y0_UM),
        ('y', '<',  ROI_Y1_UM),
    ]
    )
    
    ts(f"Transcripts: {len(pts_roi):,}")

    if len(pts_roi) > MAX_TX_PLOT:
        pts_roi = pts_roi.sample(MAX_TX_PLOT, random_state=42)
        ts(f"Subsampled to {MAX_TX_PLOT:,}")

    try:
        assigned = pts_roi['cell_id'].astype(float) != 0
    except Exception:
        assigned = pts_roi['cell_id'].astype(str) != '0'

    # ── Cellpose boundary image ────────────────────────────────────────────────
    ts("Computing Cellpose boundaries...")
    bnd_cp = find_boundaries(merged_roi, mode='inner').astype(np.float32)
    cmap_w = mcolors.ListedColormap(['none', 'white'])
    del merged_roi; gc.collect()

    # ── Plot ───────────────────────────────────────────────────────────────────
    panel_w = 10
    fig, axes = plt.subplots(
        1, 4,
        figsize=(panel_w * 4, panel_w * aspect),
        dpi=300, facecolor='black'
    )

    # Panel 1: DAPI only
    axes[0].imshow(dapi, extent=extent, cmap='gray', vmin=0, vmax=1, aspect='equal')
    draw_tile_lines(axes[0])
    style_ax(axes[0], 'DAPI')

    # Panel 2: DAPI + Cellpose
    axes[1].imshow(dapi, extent=extent, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axes[1].imshow(bnd_cp, extent=extent, cmap=cmap_w, vmin=0, vmax=1,
                   aspect='equal', interpolation='none', alpha=0.9)
    draw_tile_lines(axes[1])
    style_ax(axes[1], f'Cellpose ({n_cp:,} cells)')
    del bnd_cp; gc.collect()

    # Panel 3: DAPI + Proseg
    axes[2].imshow(dapi, extent=extent, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ts("Rendering Proseg boundaries (panel 3)...")
    draw_proseg_boundaries(axes[2], gdf_roi)
    draw_tile_lines(axes[2])
    style_ax(axes[2], f'Proseg ({len(gdf_roi):,} cells)')

    # Panel 4: DAPI + Proseg + transcripts
    axes[3].imshow(dapi, extent=extent, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ts("Rendering Proseg boundaries (panel 4)...")
    draw_proseg_boundaries(axes[3], gdf_roi, alpha=0.7)
    ts("Rendering transcripts...")
    if assigned.sum() > 0:
        axes[3].scatter(
            pts_roi.loc[assigned, 'x'], pts_roi.loc[assigned, 'y'],
            s=TX_DOT_SIZE, c='yellow', alpha=TX_ASSIGNED_ALPHA,
            linewidths=0, rasterized=True,
            label=f'Assigned ({assigned.sum():,})'
        )
    if (~assigned).sum() > 0:
        axes[3].scatter(
            pts_roi.loc[~assigned, 'x'], pts_roi.loc[~assigned, 'y'],
            s=TX_DOT_SIZE, c='red', alpha=TX_UNASSIGNED_ALPHA,
            linewidths=0, rasterized=True,
            label=f'Unassigned ({(~assigned).sum():,})'
        )
    draw_tile_lines(axes[3])
    style_ax(axes[3], 'Proseg + transcripts\nyellow=assigned  red=unassigned')
    axes[3].legend(fontsize=6, markerscale=6,
                   facecolor='#222222', labelcolor='white', loc='lower right')

    plt.suptitle(
        f'ROI — DS x[{ROI_X0_DS}:{ROI_X1_DS}] y[{ROI_Y0_DS}:{ROI_Y1_DS}]  |  '
        f'µm x[{ROI_X0_UM:.0f}:{ROI_X1_UM:.0f}] y[{ROI_Y0_UM:.0f}:{ROI_Y1_UM:.0f}]\n'
        f'Red dashed = tile core boundaries',
        color='white', fontsize=10, y=1.005
    )
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "region_of_interest.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    ts(f"Saved: {out_path}")

    ts(f"\n=== ROI Summary ===")
    ts(f"Cellpose cells:    {n_cp:,}")
    ts(f"Proseg cells:      {len(gdf_roi):,}")
    ts(f"Transcripts:       {len(pts_roi):,}")
    ts(f"  Assigned:        {assigned.sum():,} ({assigned.mean()*100:.1f}%)")
    ts(f"  Unassigned:      {(~assigned).sum():,}")

    elapsed = time.time() - t0
    ts(f"=== COMPLETE — {elapsed/60:.1f} mins ===")
    ts(f"Output: {out_path}")