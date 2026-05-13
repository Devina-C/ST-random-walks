#!/usr/bin/env python3
"""
proseg_pilot_viz.py
===================
Visualises the Proseg pilot results for the 2x2 patch region.

Three outputs:
  1. wholeslide_polygons_before.png  — raw Cellpose boundaries (merged_masks.npy)
  2. wholeslide_polygons_after.png   — Proseg-refined boundaries
  3. wholeslide_polygons_compare.png — side-by-side comparison

All overlaid on DAPI, cropped to pilot region.

Run on a compute node:
  sbatch scripts/slurm/submit_proseg_viz.sh
  (requires --mem=24G --cpus-per-task=4 --time=01:00:00)
"""

import os, gc, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import spatialdata as sd
from skimage.segmentation import find_boundaries
from skimage import measure
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

XENIUM_ZARR    = "/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr"
MERGED_NPY     = "data/merged/merged_masks.npy"
PROSEG_PARQUET = "data/proseg_pilot/merged_proseg_pilot_filtered.parquet"
OUT_DIR        = "data/proseg_pilot/viz"
os.makedirs(OUT_DIR, exist_ok=True)

# Coordinate system
DS_UM    = 0.425   # µm per DS pixel
FACTOR   = 2

# Pilot region in DS coords (patches 0,1,4,5)
PILOT_X0_DS = 0
PILOT_Y0_DS = 0
PILOT_X1_DS = 12816
PILOT_Y1_DS = 9368

# Pilot region in microns
PILOT_X0_UM = PILOT_X0_DS * DS_UM
PILOT_Y0_UM = PILOT_Y0_DS * DS_UM
PILOT_X1_UM = PILOT_X1_DS * DS_UM
PILOT_Y1_UM = PILOT_Y1_DS * DS_UM


def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}", flush=True)


# ── Load DAPI for pilot region ────────────────────────────────────────────────

def load_dapi_pilot():
    ts("Loading DAPI...")
    sdata  = sd.read_zarr(XENIUM_ZARR)
    img_dt = sdata.images[list(sdata.images.keys())[0]]

    # Load pyramid level 1 (DS resolution = full_res / 2)
    for lv in range(1, 5):
        try:
            ds_  = img_dt[f'scale{lv}'].ds
            var  = list(ds_.data_vars)[0]
            arr  = ds_[var]
            print(f"  Level {lv}: shape={arr.shape}", flush=True)
            if arr.ndim == 3:
                img = arr[0].values
            else:
                img = arr.values
            img = img.astype(np.float32)
            del sdata, arr, ds_
            break
        except Exception as e:
            print(f"  Level {lv} failed: {e}", flush=True)
            continue

    # Crop to pilot region
    img_crop = img[PILOT_Y0_DS:PILOT_Y1_DS, PILOT_X0_DS:PILOT_X1_DS]

    # Normalise to [0,1]
    nonzero = img_crop[img_crop > 0]
    if len(nonzero) > 0:
        p1, p99 = np.percentile(nonzero, [1, 99])
        img_crop = np.clip((img_crop - p1) / (p99 - p1 + 1e-6), 0, 1)

    ts(f"DAPI loaded: {img_crop.shape}")
    del img
    gc.collect()
    return img_crop


# ── Build boundary image from pixel canvas ────────────────────────────────────

def boundaries_from_masks(merged_crop):
    """Find cell boundaries from pixel label array using find_boundaries."""
    ts("  Deriving boundaries from pixel canvas...")
    bnd = find_boundaries(merged_crop, mode='inner').astype(np.uint8)
    return bnd


# ── Build boundary image from Proseg polygons ─────────────────────────────────

def boundaries_from_polygons(gdf, h, w):
    """
    Rasterise Proseg polygon boundaries onto a pixel canvas.
    Polygons are in micron space — convert to DS pixel coords.
    """
    ts("  Rasterising Proseg polygon boundaries...")
    bnd = np.zeros((h, w), dtype=np.uint8)

    img_pil  = Image.fromarray(bnd)
    draw     = ImageDraw.Draw(img_pil)

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        # Handle MultiPolygon
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]

        for poly in polys:
            if poly.is_empty:
                continue
            # Convert micron coords to DS pixel coords relative to pilot origin
            coords = [
                (
                    (x - PILOT_X0_UM) / DS_UM,
                    (y - PILOT_Y0_UM) / DS_UM
                )
                for x, y in poly.exterior.coords
            ]
            if len(coords) >= 3:
                draw.polygon(coords, outline=255, fill=None)

    bnd = np.array(img_pil)
    del img_pil, draw
    gc.collect()
    return bnd


# ── Composite DAPI + boundaries ───────────────────────────────────────────────

def composite(dapi, boundary, colour=(255, 255, 255)):
    """Overlay boundaries on greyscale DAPI."""
    rgb = np.stack([dapi, dapi, dapi], axis=2)
    rgb = (rgb * 255).astype(np.uint8)
    rgb[boundary > 0] = colour
    return rgb


# ── Draw tile grid lines ──────────────────────────────────────────────────────

def draw_tile_grid(rgb, core_size_ds=3000, colour=(255, 0, 0)):
    """Draw tile core boundaries in red."""
    h, w = rgb.shape[:2]
    x = core_size_ds
    while x < w:
        rgb[:, max(0, x-1):x+1] = colour
        x += core_size_ds
    y = core_size_ds
    while y < h:
        rgb[max(0, y-1):y+1, :] = colour
        y += core_size_ds
    return rgb


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("=== PROSEG PILOT VIZ ===")

    # Load DAPI
    dapi = load_dapi_pilot()
    h, w = dapi.shape

    # ── Before: Cellpose boundaries from merged_masks.npy ─────────────────────
    ts("Loading merged masks...")
    merged_full = np.load(MERGED_NPY, mmap_mode='r')
    merged_crop = merged_full[PILOT_Y0_DS:PILOT_Y1_DS,
                               PILOT_X0_DS:PILOT_X1_DS].copy()
    del merged_full; gc.collect()
    ts(f"Merged masks cropped: {merged_crop.shape}, "
       f"cells={np.unique(merged_crop[merged_crop>0]).shape[0]:,}")

    ts("Building Cellpose boundary image...")
    bnd_before = boundaries_from_masks(merged_crop)
    del merged_crop; gc.collect()

    rgb_before = composite(dapi, bnd_before)
    rgb_before = draw_tile_grid(rgb_before)
    out_before = os.path.join(OUT_DIR, "pilot_before_cellpose.png")
    Image.fromarray(rgb_before).save(out_before)
    ts(f"Saved: {out_before}")
    del bnd_before, rgb_before; gc.collect()

    # ── After: Proseg polygon boundaries ──────────────────────────────────────
    ts("Loading Proseg shapes...")
    gdf = gpd.read_parquet(PROSEG_PARQUET)
    ts(f"Proseg shapes: {len(gdf):,}")

    ts("Building Proseg boundary image...")
    bnd_after = boundaries_from_polygons(gdf, h, w)
    del gdf; gc.collect()

    rgb_after = composite(dapi, bnd_after)
    rgb_after = draw_tile_grid(rgb_after)
    out_after = os.path.join(OUT_DIR, "pilot_after_proseg.png")
    Image.fromarray(rgb_after).save(out_after)
    ts(f"Saved: {out_after}")

    # ── Side by side comparison ────────────────────────────────────────────────
    ts("Building comparison image...")

    # Reload before boundaries for comparison
    merged_full = np.load(MERGED_NPY, mmap_mode='r')
    merged_crop = merged_full[PILOT_Y0_DS:PILOT_Y1_DS,
                               PILOT_X0_DS:PILOT_X1_DS].copy()
    del merged_full; gc.collect()
    bnd_before = boundaries_from_masks(merged_crop)
    del merged_crop; gc.collect()

    # Before: white boundaries
    rgb_before = composite(dapi, bnd_before, colour=(255, 255, 255))
    rgb_before = draw_tile_grid(rgb_before, colour=(255, 0, 0))

    # After: cyan boundaries for contrast
    rgb_after  = composite(dapi, bnd_after,  colour=(0, 255, 255))
    rgb_after  = draw_tile_grid(rgb_after,  colour=(255, 0, 0))

    # Add dividing line between panels
    divider = np.full((h, 10, 3), 128, dtype=np.uint8)
    comparison = np.concatenate([rgb_before, divider, rgb_after], axis=1)

    out_compare = os.path.join(OUT_DIR, "pilot_comparison.png")
    Image.fromarray(comparison).save(out_compare)
    ts(f"Saved: {out_compare}")
    del comparison; gc.collect()

    # ── High-DPI polygon viz for close inspection ─────────────────────────────
    ts("Building high-DPI polygon viz...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from shapely.geometry import box as shapely_box
    import shapely

    # Reload Proseg shapes
    gdf = gpd.read_parquet(PROSEG_PARQUET)

    # Pick a representative zoom region — centre of pilot area
    # ~2000x2000 DS px around the middle (tile boundary region)
    zoom_cx_ds = PILOT_X1_DS // 2   # 6408
    zoom_cy_ds = PILOT_Y1_DS // 2   # 4684
    zoom_half  = 1000                # ±1000 DS px = 2000x2000 region

    zoom_x0_ds = zoom_cx_ds - zoom_half
    zoom_x1_ds = zoom_cx_ds + zoom_half
    zoom_y0_ds = zoom_cy_ds - zoom_half
    zoom_y1_ds = zoom_cy_ds + zoom_half

    zoom_x0_um = zoom_x0_ds * DS_UM
    zoom_x1_um = zoom_x1_ds * DS_UM
    zoom_y0_um = zoom_y0_ds * DS_UM
    zoom_y1_um = zoom_y1_ds * DS_UM

    # Crop DAPI to zoom region
    dapi_zoom = dapi[zoom_y0_ds:zoom_y1_ds, zoom_x0_ds:zoom_x1_ds]

    # Filter shapes to zoom region
    zoom_box  = shapely_box(zoom_x0_um, zoom_y0_um, zoom_x1_um, zoom_y1_um)
    gdf_zoom  = gdf[gdf.intersects(zoom_box)].copy()
    ts(f"  Shapes in zoom region: {len(gdf_zoom):,}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=2000, facecolor='black')

    for ax, title, colour in zip(
        axes,
        ['Cellpose (before)', 'Proseg (after)'],
        ['white', 'cyan']
    ):
        ax.set_facecolor('black')
        ax.imshow(
            dapi_zoom,
            extent=[zoom_x0_um, zoom_x1_um, zoom_y1_um, zoom_y0_um],
            cmap='gray', vmin=0, vmax=1, aspect='equal'
        )
        ax.set_xlim(zoom_x0_um, zoom_x1_um)
        ax.set_ylim(zoom_y1_um, zoom_y0_um)
        ax.set_title(title, color='white', fontsize=14, pad=8)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    # Before: Cellpose boundaries from merged_masks crop
    ax = axes[0]
    merged_full = np.load(MERGED_NPY, mmap_mode='r')
    merged_zoom = merged_full[zoom_y0_ds:zoom_y1_ds, zoom_x0_ds:zoom_x1_ds].copy()
    del merged_full; gc.collect()

    bnd_zoom = find_boundaries(merged_zoom, mode='inner').astype(np.float32)

    # Overlay as imshow with white colormap, transparent where no boundary
    import matplotlib.colors as mcolors
    cmap_bnd = mcolors.ListedColormap(['none', 'white'])
    ax.imshow(
        bnd_zoom,
        extent=[zoom_x0_um, zoom_x1_um, zoom_y1_um, zoom_y0_um],
        cmap=cmap_bnd, vmin=0, vmax=1,
        aspect='equal', interpolation='none', alpha=0.9
    )
    del merged_zoom, bnd_zoom; gc.collect()

    # After: Proseg polygons
    ax = axes[1]
    for geom in gdf_zoom.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            if poly.is_empty: continue
            xs_p, ys_p = poly.exterior.xy
            ax.plot(xs_p, ys_p, color='cyan', linewidth=0.3, alpha=0.8)

    # Draw tile boundaries on both panels
    for ax in axes:
        # Vertical tile lines
        x = 3000 * DS_UM
        while x < PILOT_X1_DS * DS_UM:
            if zoom_x0_um <= x <= zoom_x1_um:
                ax.axvline(x, color='red', linewidth=0.8, alpha=0.7, linestyle='--')
            x += 3000 * DS_UM
        # Horizontal tile lines
        y = 3000 * DS_UM
        while y < PILOT_Y1_DS * DS_UM:
            if zoom_y0_um <= y <= zoom_y1_um:
                ax.axhline(y, color='red', linewidth=0.8, alpha=0.7, linestyle='--')
            y += 3000 * DS_UM

    plt.suptitle(
        f'Proseg Pilot — Zoom region DS x[{zoom_x0_ds}:{zoom_x1_ds}] '
        f'y[{zoom_y0_ds}:{zoom_y1_ds}]\n'
        f'Red dashed = tile boundaries',
        color='white', fontsize=12, y=0.98
    )
    plt.tight_layout()

    out_highdpi = os.path.join(OUT_DIR, "pilot_highdpi_comparison.png")
    plt.savefig(out_highdpi, dpi=2000, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    ts(f"Saved: {out_highdpi}")

    del gdf, gdf_zoom, dapi_zoom; gc.collect()

    elapsed = time.time() - t0
    ts(f"=== COMPLETE — {elapsed/60:.1f} mins ===")
    ts(f"Outputs in: {OUT_DIR}/")
    ts(f"  pilot_before_cellpose.png   — raw Cellpose boundaries (white) + tile grid (red)")
    ts(f"  pilot_after_proseg.png      — Proseg-refined boundaries (white) + tile grid (red)")
    ts(f"  pilot_comparison.png        — side by side pixel-level (before=white | after=cyan)")
    ts(f"  pilot_highdpi_comparison.png — 2000 DPI polygon overlay on DAPI, zoom region")