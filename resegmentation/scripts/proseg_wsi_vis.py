#!/usr/bin/env python3
"""
proseg_wsi_viz.py
=================
Whole-slide visualisation of Proseg-refined cell boundaries overlaid on DAPI.

Two outputs:
  data/proseg_wsi/viz/wsi_proseg_boundaries.png  — polygon outlines on DAPI
  data/proseg_wsi/viz/wsi_proseg_pixels.png      — random-colour fill per cell

Polygon rasterisation is done via PIL — 1 DS pixel = 1 image pixel,
so the output is (37473 x 25633) pixels. Open in any viewer and zoom in.

Run on a compute node:
  sbatch scripts/slurm/submit_proseg_wsi_viz.sh
  (requires --mem=48G --cpus-per-task=4 --time=04:00:00)
"""

import os, gc, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import spatialdata as sd
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

XENIUM_ZARR    = "/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr"
PROSEG_PARQUET = "data/proseg_wsi/merged_proseg_wsi.parquet"
OUT_DIR        = "data/proseg_wsi/viz"
os.makedirs(OUT_DIR, exist_ok=True)

DS_UM    = 0.425
GLOBAL_H = 37473
GLOBAL_W = 25633

# Original 32-patch grid for reference lines
N_COLS, N_ROWS = 4, 8
PATCH_W = 51265 / N_COLS
PATCH_H = 74945 / N_ROWS
FACTOR  = 2


def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}", flush=True)


# ── Load DAPI full slide ──────────────────────────────────────────────────────

def load_dapi():
    ts("Loading full-slide DAPI...")
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
        except Exception as e:
            print(f"  Level {lv} failed: {e}", flush=True)
            continue
    nonzero = img[img > 0]
    if len(nonzero):
        p1, p99 = np.percentile(nonzero, [1, 99])
        img     = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    ts(f"DAPI: {img.shape}")
    gc.collect()
    return img


# ── Polygon boundary viz ──────────────────────────────────────────────────────

def make_polygon_viz(dapi, gdf):
    """
    Rasterise Proseg polygon outlines onto DAPI background.
    Polygons are in micron space — convert to DS pixel coords.
    """
    ts("Building polygon boundary viz...")
    h, w = dapi.shape[:2]

    # Build RGB DAPI background
    dapi_8 = (dapi * 255).astype(np.uint8)
    rgb    = np.stack([dapi_8, dapi_8, dapi_8], axis=2)
    del dapi_8; gc.collect()

    # Draw polygons onto PIL image
    img_pil = Image.fromarray(rgb)
    draw    = ImageDraw.Draw(img_pil)
    del rgb; gc.collect()

    ts(f"  Drawing {len(gdf):,} cell polygons...")
    n_drawn = 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            if poly.is_empty: continue
            coords = [
                (x / DS_UM, y / DS_UM) 
                for x, y in poly.exterior.coords
            ]
            if len(coords) >= 3:
                draw.polygon(coords, outline=(255, 255, 255), fill=None)
        n_drawn += 1

    ts(f"  Drew {n_drawn:,} cells")

    # Draw patch grid in red
    for row in range(N_ROWS + 1):
        y_ds = int(row * PATCH_H // FACTOR)
        if 0 <= y_ds < h:
            draw.line([(0, y_ds), (w, y_ds)], fill=(255, 0, 0), width=2)
    for col in range(N_COLS + 1):
        x_ds = int(col * PATCH_W // FACTOR)
        if 0 <= x_ds < w:
            draw.line([(x_ds, 0), (x_ds, h)], fill=(255, 0, 0), width=2)

    out = os.path.join(OUT_DIR, "wsi_proseg_boundaries.png")
    img_pil.save(out)
    ts(f"Saved: {out}")
    del img_pil, draw; gc.collect()


# ── Random colour fill viz ────────────────────────────────────────────────────

def make_colour_fill_viz(gdf):
    """
    Rasterise filled Proseg polygons with random colours.
    Each cell gets a unique random colour — useful for checking
    cell identity at tile boundaries.
    """
    ts("Building colour fill viz...")

    rng    = np.random.default_rng(42)
    n_cells = len(gdf)
    colours = rng.integers(40, 220, size=(n_cells, 3), dtype=np.uint8)

    img_pil = Image.new('RGB', (GLOBAL_W, GLOBAL_H), (0, 0, 0))
    draw    = ImageDraw.Draw(img_pil)

    ts(f"  Filling {n_cells:,} cells...")
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        colour = tuple(colours[i].tolist())
        polys  = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            if poly.is_empty: continue
            coords = [
                (x / DS_UM / 2, y / DS_UM / 2)
                for x, y in poly.exterior.coords
            ]
            if len(coords) >= 3:
                draw.polygon(coords, fill=colour, outline=None)

        if (i + 1) % 50000 == 0:
            ts(f"  {i+1:,}/{n_cells:,} cells drawn...")

    # Patch grid
    h, w = GLOBAL_H, GLOBAL_W
    for row in range(N_ROWS + 1):
        y_ds = int(row * PATCH_H // FACTOR)
        if 0 <= y_ds < h:
            draw.line([(0, y_ds), (w, y_ds)], fill=(255, 255, 255), width=2)
    for col in range(N_COLS + 1):
        x_ds = int(col * PATCH_W // FACTOR)
        if 0 <= x_ds < w:
            draw.line([(x_ds, 0), (x_ds, h)], fill=(255, 255, 255), width=2)

    out = os.path.join(OUT_DIR, "wsi_proseg_pixels.png")
    img_pil.save(out)
    ts(f"Saved: {out}")
    del img_pil, draw; gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("=== PROSEG WSI VIZ ===")

    ts("Loading Proseg shapes...")
    gdf = gpd.read_parquet(PROSEG_PARQUET)
    ts(f"Loaded {len(gdf):,} cells")

    # Polygon boundary viz (needs DAPI)
    dapi = load_dapi()
    make_polygon_viz(dapi, gdf)
    del dapi; gc.collect()

    # Colour fill viz (no DAPI needed)
    #make_colour_fill_viz(gdf)

    elapsed = time.time() - t0
    ts(f"=== COMPLETE — {elapsed/3600:.2f}h ===")
    ts(f"Outputs in: {OUT_DIR}/")
    ts(f"  wsi_proseg_boundaries.png — white polygon outlines on DAPI")
    #ts(f"  wsi_proseg_pixels.png     — random colour fills per cell")