#!/usr/bin/env python3
"""
reassemble_proseg_wsi.py
========================
Reassembles the per-tile Proseg outputs into a single global parquet,
applying the transcript count filter correctly.

What went wrong in the original run:
  The assembly step looked for adata.obs['transcript_count'] which doesn't
  exist in Proseg's output. Proseg stores transcript counts in the expression
  matrix adata.X (cell x gene), not in obs metadata. The fix is to compute
  counts as adata.X.sum(axis=1) — the row sum of the expression matrix gives
  total transcripts assigned to each cell.

Proseg output structure per tile:
  tile_RxC/proseg_out.zarr/
    shapes/cell_boundaries/shapes.parquet  ← polygon boundaries
    tables/table/                          ← AnnData zarr
      obs/   ← cell metadata (centroid, volume, surface_area etc)
      var/   ← gene names
      X/     ← sparse cell x gene count matrix ← transcript counts are HERE

Filters applied:
  1. Shape quality: area, convexity, aspect ratio (same as original)
  2. Transcript count >= MIN_TRANSCRIPTS (from adata.X.sum(axis=1))

Output:
  data/proseg_wsi/merged_proseg_wsi.parquet      — filtered global shapes
  data/proseg_wsi/merged_proseg_wsi_stats.csv    — per-cell stats
"""

import os, gc, glob, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import anndata as ad
from shapely.validation import make_valid
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR       = "data/proseg_wsi"
MIN_TX        = 5        # minimum transcripts per cell
MIN_AREA_UM2  = 10.0    # µm² — remove tiny fragments
MAX_AREA_UM2  = 2000.0  # µm² — remove giant artefacts
MIN_CONVEXITY = 0.3     # convexity = area / convex_hull_area


def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}", flush=True)


# ── Load transcript counts from Proseg zarr ───────────────────────────────────

def load_transcript_counts(tile_name):
    """
    Load per-cell transcript counts from Proseg's AnnData zarr.

    Proseg stores the cell x gene expression matrix in adata.X.
    Row sums give total transcripts assigned to each cell by Proseg.
    This is the correct source — NOT adata.obs which only has geometry.

    Returns numpy array of counts, one per cell, or None if unavailable.
    """
    zarr_path = os.path.join(OUT_DIR, tile_name,
                             "proseg_out.zarr", "tables", "table")
    if not os.path.exists(zarr_path):
        return None
    try:
        adata  = ad.read_zarr(zarr_path)
        counts = np.array(adata.X.sum(axis=1)).flatten()
        return counts
    except Exception as e:
        print(f"    WARNING: could not load adata for {tile_name}: {e}")
        return None


# ── Shape quality filter ──────────────────────────────────────────────────────

def apply_shape_filters(gdf):
    """
    Filter out geometrically implausible shapes:
      - Too small (fragments from Proseg artefacts)
      - Too large (flow convergence artefacts)
      - Too non-convex (spider/flower shapes)
    """
    before = len(gdf)

    # Ensure valid geometries
    gdf['geometry'] = gdf['geometry'].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    gdf = gdf[gdf['geometry'].notna() & ~gdf['geometry'].is_empty].copy()

    # Area in µm² (polygon coords are in µm)
    gdf['area_um2'] = gdf['geometry'].area
    gdf = gdf[(gdf['area_um2'] >= MIN_AREA_UM2) &
              (gdf['area_um2'] <= MAX_AREA_UM2)].copy()

    # Convexity = area / convex hull area
    gdf['convexity'] = gdf['geometry'].area / \
                       gdf['geometry'].convex_hull.area.clip(lower=1e-6)
    gdf = gdf[gdf['convexity'] >= MIN_CONVEXITY].copy()

    print(f"    Shape filter: {before} → {len(gdf)} "
          f"({before - len(gdf)} removed)")
    return gdf


# ── Main assembly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("=== REASSEMBLE PROSEG WSI ===")

    shape_files = sorted(glob.glob(
        os.path.join(OUT_DIR, "shapes", "tile_*.parquet")))
    ts(f"Found {len(shape_files)} tile parquet files")

    gdfs           = []
    total_before   = 0
    total_shape_rm = 0
    total_tx_rm    = 0

    for sf in shape_files:
        tile_name = os.path.basename(sf).replace(".parquet", "")
        gdf_tile  = gpd.read_parquet(sf)

        if len(gdf_tile) == 0:
            continue

        total_before += len(gdf_tile)

        # ── Shape quality filter ───────────────────────────────────────────────
        n_before_shape = len(gdf_tile)
        gdf_tile = apply_shape_filters(gdf_tile)
        total_shape_rm += n_before_shape - len(gdf_tile)

        if len(gdf_tile) == 0:
            continue

        # ── Transcript count filter ────────────────────────────────────────────
        counts = load_transcript_counts(tile_name)

        if counts is not None:
            # Align counts to shape rows by position
            # Proseg outputs shapes in same order as adata rows
            gdf_tile  = gdf_tile.reset_index(drop=True)
            n_aligned = min(len(counts), len(gdf_tile))
            gdf_tile['transcript_count'] = np.nan
            gdf_tile.loc[:n_aligned-1, 'transcript_count'] = \
                counts[:n_aligned]

            n_before_tx = len(gdf_tile)
            gdf_tile    = gdf_tile[
                gdf_tile['transcript_count'] >= MIN_TX].copy()
            total_tx_rm += n_before_tx - len(gdf_tile)
            ts(f"  {tile_name}: {n_before_tx} → {len(gdf_tile)} "
               f"(tx filter removed {n_before_tx - len(gdf_tile)})")
        else:
            ts(f"  {tile_name}: no transcript counts — skipping tx filter")
            gdf_tile['transcript_count'] = np.nan

        gdfs.append(gdf_tile)
        del gdf_tile; gc.collect()

    # ── Concatenate ────────────────────────────────────────────────────────────
    ts("Concatenating tiles...")
    merged = pd.concat(gdfs, ignore_index=True)
    del gdfs; gc.collect()

    # Assign globally unique cell IDs
    merged['cell_id'] = [f"proseg_{i+1}" for i in range(len(merged))]

    # ── Summary stats ──────────────────────────────────────────────────────────
    ts(f"\n=== Assembly summary ===")
    ts(f"Cells before any filter:      {total_before:,}")
    ts(f"Removed by shape filter:      {total_shape_rm:,}")
    ts(f"Removed by transcript filter: {total_tx_rm:,}")
    ts(f"Final cell count:             {len(merged):,}")

    if 'area_um2' in merged.columns:
        ts(f"Area µm²:   mean={merged['area_um2'].mean():.1f}, "
           f"median={merged['area_um2'].median():.1f}")
    if 'transcript_count' in merged.columns:
        tc = merged['transcript_count'].dropna()
        ts(f"Transcripts: mean={tc.mean():.1f}, "
           f"median={tc.median():.1f}, max={tc.max():.0f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_parquet = os.path.join(OUT_DIR, "merged_proseg_wsi.parquet")
    merged.to_parquet(out_parquet)
    ts(f"Saved shapes → {out_parquet}")

    # Save stats CSV
    stats_cols  = ['cell_id', 'transcript_count', 'area_um2', 'convexity',
                   'tile_row', 'tile_col']
    stats_cols  = [c for c in stats_cols if c in merged.columns]
    stats_csv   = os.path.join(OUT_DIR, "merged_proseg_wsi_stats.csv")
    merged[stats_cols].to_csv(stats_csv, index=False)
    ts(f"Saved stats  → {stats_csv}")

    elapsed = time.time() - t0
    ts(f"=== COMPLETE — {elapsed/60:.1f} mins ===")