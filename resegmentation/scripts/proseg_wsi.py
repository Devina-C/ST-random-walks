#!/usr/bin/env python3
"""
proseg_wsi.py
=============
Whole-slide Proseg tiling run on the full BC_prime slide.

Scales proseg_pilot.py from the 2x2 patch pilot region to the full slide.
Uses the globally merged Cellpose pixel canvas (merged_masks.npy) as prior
segmentation and the raw Xenium transcripts as evidence.

Full slide dimensions (DS coords):
  x: 0 → 25633  (full width)
  y: 0 → 37473  (full height)

Tiling:
  Core size:  3000 DS px = 1275 um
  Halo:        300 DS px = 127.5 um
  Grid: ceil(25633/3000) x ceil(37473/3000) = 9 cols x 13 rows = 117 tiles

Filtering is deferred to the LAST step (assemble_tiles).
All cells are kept through tiling and stitching.
Shape quality filters (convexity, area, aspect ratio) applied at end.
Transcript count filter applied at end.

Coordinate systems:
  DS pixel:   merged_masks.npy space (full_res // 2)
  Full_res:   DS * 2
  Micron:     full_res * 0.2125  →  DS * 0.425

Estimated runtime: ~26 hours (117 tiles at ~13 min/tile)
Request: --time=36:00:00 --mem=128G --cpus-per-task=8

Run on a compute node:
  sbatch scripts/slurm/submit_proseg_wsi.sh
"""

import os, gc, gzip, glob, shutil, subprocess, time, warnings, math
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shapely_box
from scipy.stats import median_abs_deviation
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

PROSEG_BIN      = "/users/k22026807/.cargo/bin/proseg"
TRANSCRIPTS_PAR = ("/scratch/users/k22026807/masters/project/xenium_output/"
                   "BC_prime.zarr/points/transcripts/points.parquet")
MERGED_NPY      = "data/merged/merged_masks.npy"
OUT_DIR         = "data/proseg_wsi"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/shapes", exist_ok=True)
os.makedirs(f"{OUT_DIR}/logs",   exist_ok=True)

# Coordinate system
PIXEL_SIZE_UM = 0.2125
DS_FACTOR     = 2
DS_UM         = PIXEL_SIZE_UM * DS_FACTOR   # 0.425 µm per DS pixel

# Full slide dimensions in DS coords
WSI_X_DS = 25633
WSI_Y_DS = 37473

# Tiling — same as pilot
CORE_DS  = 3000   # core size in DS px = 1275 µm
HALO_DS  = 300    # halo in DS px = 127.5 µm

# Proseg parameters — same as pilot
PROSEG_SAMPLES        = "1000"
PROSEG_BURNIN_SAMPLES = "100"
PROSEG_VOXEL_SIZE     = "0.425"
PROSEG_BURNIN_SIZE    = "0.850"
PROSEG_VOXEL_LAYERS   = "2"
PROSEG_NUCLEAR_PROB   = "0.25"
PROSEG_DIFF_PROB      = "0.25"
PROSEG_NTHREADS       = "8"

# Shape quality filter thresholds — applied LAST in assemble_tiles
MIN_AREA_UM2     = 4.0      # physically impossible minimum
MAX_AREA_UM2     = 10000.0  # physically impossible maximum
MIN_CONVEXITY    = 0.7      # removes flower/rosette artefacts
MAX_ASPECT_RATIO = 10.0     # removes boundary slivers
MIN_TRANSCRIPTS  = 5        # minimum transcripts per cell


# ── Timestamp helper ──────────────────────────────────────────────────────────

def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}", flush=True)


# ── DS coords → microns ───────────────────────────────────────────────────────

def ds_to_um(ds_val):
    return ds_val * DS_UM


# ── Build tile grid ───────────────────────────────────────────────────────────

def build_tile_grid(wsi_x, wsi_y, core_size, halo_size):
    """
    Build list of tile dicts covering the full slide.
    core_bounds: (x0, y0, x1, y1) — strict property lines (half-open)
    tile_bounds: (x0, y0, x1, y1) — expanded by halo for Proseg
    """
    n_cols = math.ceil(wsi_x / core_size)
    n_rows = math.ceil(wsi_y / core_size)
    tiles  = []

    for row in range(n_rows):
        for col in range(n_cols):
            cx0 = col * core_size
            cx1 = min((col + 1) * core_size, wsi_x)
            cy0 = row * core_size
            cy1 = min((row + 1) * core_size, wsi_y)

            tx0 = max(0,     cx0 - halo_size)
            tx1 = min(wsi_x, cx1 + halo_size)
            ty0 = max(0,     cy0 - halo_size)
            ty1 = min(wsi_y, cy1 + halo_size)

            tiles.append({
                'row': row, 'col': col,
                'core': (cx0, cy0, cx1, cy1),
                'tile': (tx0, ty0, tx1, ty1),
            })

    print(f"Tile grid: {n_cols} cols x {n_rows} rows = {len(tiles)} tiles")
    return tiles, n_cols, n_rows


# ── Crop transcripts to tile ──────────────────────────────────────────────────

def crop_transcripts(pts_df, tx0_um, ty0_um, tx1_um, ty1_um):
    mask = (
        (pts_df['x'] >= tx0_um) & (pts_df['x'] < tx1_um) &
        (pts_df['y'] >= ty0_um) & (pts_df['y'] < ty1_um)
    )
    return pts_df[mask].copy()


# ── Crop merged masks to tile ─────────────────────────────────────────────────

def crop_masks(merged, tx0, ty0, tx1, ty1):
    crop = merged[ty0:ty1, tx0:tx1].copy()
    uids = np.unique(crop); uids = uids[uids > 0]
    if len(uids) == 0:
        return crop
    remap = np.zeros(int(crop.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(uids, start=1):
        remap[old_id] = new_id
    return remap[crop]


# ── Save mask as gzipped npy ──────────────────────────────────────────────────

def save_npy_gz(arr, path):
    tmp = path.replace('.npy.gz', '.npy')
    np.save(tmp, arr)
    with open(tmp, 'rb') as f_in, gzip.open(path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(tmp)


# ── Check if tile already completed ──────────────────────────────────────────

def tile_already_done(row, col):
    """Check if tile parquet already exists — allows resuming interrupted runs."""
    out_parquet = os.path.join(OUT_DIR, "shapes", f"tile_{row}x{col}.parquet")
    return os.path.exists(out_parquet)


# ── Run Proseg on one tile ────────────────────────────────────────────────────

def run_proseg_tile(tile, merged, pts_df, tile_idx, n_tiles):
    row, col   = tile['row'], tile['col']
    tx0, ty0, tx1, ty1 = tile['tile']
    cx0, cy0, cx1, cy1 = tile['core']

    # Resume support — skip completed tiles
    if tile_already_done(row, col):
        ts(f"  Tile {tile_idx+1}/{n_tiles} (r{row}c{col}): already done — skipping")
        return os.path.join(OUT_DIR, "shapes", f"tile_{row}x{col}.parquet")

    ts(f"  Tile {tile_idx+1}/{n_tiles} (r{row}c{col}): "
       f"tile DS x[{tx0}:{tx1}] y[{ty0}:{ty1}], "
       f"core DS x[{cx0}:{cx1}] y[{cy0}:{cy1}]")

    tx0_um = ds_to_um(tx0); tx1_um = ds_to_um(tx1)
    ty0_um = ds_to_um(ty0); ty1_um = ds_to_um(ty1)

    # Crop transcripts
    tx_crop = crop_transcripts(pts_df, tx0_um, ty0_um, tx1_um, ty1_um)
    if len(tx_crop) == 0:
        ts(f"    No transcripts in tile — skipping")
        return None
    ts(f"    Transcripts: {len(tx_crop):,}")

    # Crop masks
    mask_crop = crop_masks(merged, tx0, ty0, tx1, ty1)
    n_cells_in_mask = np.unique(mask_crop[mask_crop > 0]).shape[0]
    ts(f"    Mask cells: {n_cells_in_mask}")

    # Remap transcript cell_id to merged Cellpose IDs
    tx_crop = tx_crop.copy()
    col_px  = ((tx_crop["x"].values / DS_UM) - tx0).astype(int).clip(
                0, mask_crop.shape[1] - 1)
    row_px  = ((tx_crop["y"].values / DS_UM) - ty0).astype(int).clip(
                0, mask_crop.shape[0] - 1)
    tx_crop["cell_id"] = mask_crop[row_px, col_px]
    n_assigned   = (tx_crop["cell_id"] > 0).sum()
    n_unassigned = (tx_crop["cell_id"] == 0).sum()
    ts(f"    Assigned to merged cells: {n_assigned:,} | unassigned: {n_unassigned:,}")

    # Working directory
    tile_dir      = os.path.join(OUT_DIR, f"tile_{row}x{col}")
    os.makedirs(tile_dir, exist_ok=True)

    tx_csv_path   = os.path.join(tile_dir, "transcripts.csv")
    mask_npy_path = os.path.join(tile_dir, "cellpose_masks.npy.gz")
    out_zarr_path = os.path.join(tile_dir, "proseg_out.zarr")

    # Save transcripts and masks
    tx_out = tx_crop[["x", "y", "z", "feature_name", "cell_id"]].copy()
    tx_out.to_csv(tx_csv_path, index=False)
    save_npy_gz(mask_crop.astype(np.uint32), mask_npy_path)
    del mask_crop; gc.collect()

    # Proseg affine transforms
    x_offset_um = ds_to_um(tx0)
    y_offset_um = ds_to_um(ty0)

    cmd = [
        PROSEG_BIN, tx_csv_path,
        "-x", "x",
        "-y", "y",
        "-z", "z",
        "--gene-column",           "feature_name",
        "--cell-id-column",        "cell_id",
        "--cell-id-unassigned",    "0",
        "--cellpose-masks",        mask_npy_path,
        "--cellpose-x-transform",  str(DS_UM), "0", str(x_offset_um),
        "--cellpose-y-transform",  "0", str(DS_UM), str(y_offset_um),
        "--samples",               PROSEG_SAMPLES,
        "--burnin-samples",        PROSEG_BURNIN_SAMPLES,
        "--voxel-size",            PROSEG_VOXEL_SIZE,
        "--burnin-voxel-size",     PROSEG_BURNIN_SIZE,
        "--voxel-layers",          PROSEG_VOXEL_LAYERS,
        "--nuclear-reassignment-prob", PROSEG_NUCLEAR_PROB,
        "--diffusion-probability",     PROSEG_DIFF_PROB,
        "--nthreads",              PROSEG_NTHREADS,
        "--output-spatialdata",    out_zarr_path,
        "--overwrite",
    ]

    # Log proseg stderr to per-tile log file
    log_path = os.path.join(OUT_DIR, "logs", f"tile_{row}x{col}.log")
    ts(f"    Running Proseg... (log: {log_path})")
    t_tile = time.time()

    with open(log_path, 'w') as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    tile_elapsed = time.time() - t_tile
    if result.returncode != 0:
        ts(f"    ERROR: Proseg failed for tile {tile_idx+1} "
           f"(r{row}c{col}) after {tile_elapsed:.0f}s — see {log_path}")
        return None
    ts(f"    Proseg complete in {tile_elapsed:.0f}s")

    # Load refined shapes
    shapes_parquet = os.path.join(
        out_zarr_path, "shapes", "cell_boundaries", "shapes.parquet")
    if not os.path.exists(shapes_parquet):
        ts(f"    WARNING: no shapes parquet found")
        return None

    gdf = gpd.read_parquet(shapes_parquet)
    ts(f"    Raw shapes from Proseg: {len(gdf):,}")

    # Centroid rule — keep only cells whose centroid is in the core region
    cx0_um = ds_to_um(cx0); cx1_um = ds_to_um(cx1)
    cy0_um = ds_to_um(cy0); cy1_um = ds_to_um(cy1)

    centroids = gdf.geometry.centroid
    keep_mask = (
        (centroids.x >= cx0_um) & (centroids.x < cx1_um) &
        (centroids.y >= cy0_um) & (centroids.y < cy1_um)
    )
    gdf_kept = gdf[keep_mask].copy()
    gdf_kept['tile_row'] = row
    gdf_kept['tile_col'] = col
    ts(f"    Kept after centroid rule: {len(gdf_kept):,} / {len(gdf):,}")

    # Load transcript counts from tile zarr — store raw counts, filter later
    try:
        import anndata as ad
        adata_tile = ad.read_zarr(
            os.path.join(OUT_DIR, f"tile_{row}x{col}",
                         "proseg_out.zarr", "tables", "table"))
        count_col = ('transcript_count' if 'transcript_count' in adata_tile.obs.columns
                     else 'cell_count'  if 'cell_count'       in adata_tile.obs.columns
                     else None)
        if count_col:
            counts = adata_tile.obs[count_col].reset_index(drop=True)
            gdf_kept = gdf_kept.reset_index(drop=True)
            # Align by position — both indexed same way after centroid filter
            # Use inner index from original gdf position
            orig_idx = np.where(keep_mask)[0]
            gdf_kept['transcript_count'] = counts.iloc[orig_idx].values
        else:
            ts(f"    WARNING: no transcript_count column — will fill NaN")
            gdf_kept['transcript_count'] = np.nan
        del adata_tile
    except Exception as e:
        ts(f"    WARNING: could not load transcript counts ({e})")
        gdf_kept['transcript_count'] = np.nan

    # Save tile shapes — NO filtering yet
    out_parquet = os.path.join(OUT_DIR, "shapes", f"tile_{row}x{col}.parquet")
    gdf_kept.to_parquet(out_parquet)
    ts(f"    Saved: {out_parquet} ({len(gdf_kept):,} cells)")

    # Cleanup tile inputs
    if os.path.exists(tx_csv_path):
        os.remove(tx_csv_path)
    if os.path.exists(mask_npy_path):
        os.remove(mask_npy_path)

    del tx_crop, gdf, gdf_kept; gc.collect()
    return out_parquet


# ── Assemble all tile outputs ─────────────────────────────────────────────────

def assemble_tiles():
    """
    Concatenate all tile outputs.
    Filtering is applied LAST — in this order:
      1. Shape quality: convexity, area, aspect ratio
      2. Transcript count: >= MIN_TRANSCRIPTS

    All cells are retained through tiling — nothing is dropped until here.
    """
    shape_files = sorted(glob.glob(
        os.path.join(OUT_DIR, "shapes", "tile_*.parquet")))
    if not shape_files:
        ts("No tile outputs found to assemble")
        return None

    ts(f"Assembling {len(shape_files)} tile outputs...")
    gdfs = []
    for sf in shape_files:
        gdf_tile = gpd.read_parquet(sf)
        gdfs.append(gdf_tile)

    merged_gdf = pd.concat(gdfs, ignore_index=True)
    del gdfs; gc.collect()
    ts(f"Total cells before filtering: {len(merged_gdf):,}")

    # ── Filter 1: Shape quality ───────────────────────────────────────────────
    ts("Applying shape quality filters...")

    merged_gdf['area_um2']     = merged_gdf.geometry.area
    merged_gdf['convexity']    = (merged_gdf.geometry.area /
                                  merged_gdf.geometry.convex_hull.area)
    bounds                     = merged_gdf.geometry.bounds
    merged_gdf['bbox_w']       = bounds['maxx'] - bounds['minx']
    merged_gdf['bbox_h']       = bounds['maxy'] - bounds['miny']
    merged_gdf['aspect_ratio'] = (
        merged_gdf[['bbox_w','bbox_h']].max(axis=1) /
        merged_gdf[['bbox_w','bbox_h']].min(axis=1).clip(lower=0.1))

    n_before = len(merged_gdf)
    shape_mask = (
        (merged_gdf['area_um2']     >= MIN_AREA_UM2)     &
        (merged_gdf['area_um2']     <= MAX_AREA_UM2)     &
        (merged_gdf['convexity']    >= MIN_CONVEXITY)    &
        (merged_gdf['aspect_ratio'] <= MAX_ASPECT_RATIO)
    )
    merged_gdf = merged_gdf[shape_mask].copy()
    ts(f"  After shape filter: {len(merged_gdf):,} "
       f"({n_before - len(merged_gdf):,} removed, "
       f"{(n_before-len(merged_gdf))/n_before*100:.1f}%)")

    # Report shape metric distributions
    ts(f"  Area:         mean={merged_gdf['area_um2'].mean():.1f} µm², "
       f"median={merged_gdf['area_um2'].median():.1f} µm²")
    ts(f"  Convexity:    mean={merged_gdf['convexity'].mean():.3f}")
    ts(f"  Aspect ratio: mean={merged_gdf['aspect_ratio'].mean():.2f}")

    # ── Filter 2: Transcript count ────────────────────────────────────────────
    ts(f"Applying transcript count filter (>={MIN_TRANSCRIPTS})...")
    n_before = len(merged_gdf)

    if 'transcript_count' in merged_gdf.columns:
        has_count  = merged_gdf['transcript_count'].notna()
        count_mask = (
            merged_gdf['transcript_count'].fillna(0) >= MIN_TRANSCRIPTS)
        merged_gdf = merged_gdf[count_mask].copy()
        ts(f"  After transcript filter: {len(merged_gdf):,} "
           f"({n_before - len(merged_gdf):,} removed)")
        ts(f"  Transcript count: "
           f"mean={merged_gdf['transcript_count'].mean():.1f}, "
           f"median={merged_gdf['transcript_count'].median():.0f}, "
           f"max={merged_gdf['transcript_count'].max():.0f}")
    else:
        ts("  WARNING: no transcript_count column — skipping count filter")

    # Drop helper columns before saving
    drop_cols = ['bbox_w', 'bbox_h']
    merged_gdf = merged_gdf.drop(
        columns=[c for c in drop_cols if c in merged_gdf.columns])

    # Assign globally unique contiguous cell IDs
    merged_gdf = merged_gdf.reset_index(drop=True)
    merged_gdf['cell_id'] = [f"proseg_{i+1}" for i in range(len(merged_gdf))]

    ts(f"Final cell count: {len(merged_gdf):,}")

    # Save filtered result
    out_path = os.path.join(OUT_DIR, "merged_proseg_wsi.parquet")
    merged_gdf.to_parquet(out_path)
    ts(f"Saved → {out_path}")

    # Save cell stats summary
    stats = pd.DataFrame({
        'n_cells':          [len(merged_gdf)],
        'mean_area_um2':    [merged_gdf['area_um2'].mean()],
        'median_area_um2':  [merged_gdf['area_um2'].median()],
        'mean_convexity':   [merged_gdf['convexity'].mean()],
        'mean_tx_count':    [merged_gdf['transcript_count'].mean()
                             if 'transcript_count' in merged_gdf.columns
                             else np.nan],
        'median_tx_count':  [merged_gdf['transcript_count'].median()
                             if 'transcript_count' in merged_gdf.columns
                             else np.nan],
    })
    stats.to_csv(os.path.join(OUT_DIR, "cell_stats.csv"), index=False)
    ts(f"Saved → {OUT_DIR}/cell_stats.csv")

    return merged_gdf


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("=== PROSEG WSI START ===")
    ts(f"Full slide: {WSI_X_DS} x {WSI_Y_DS} DS px "
       f"= {ds_to_um(WSI_X_DS):.0f} x {ds_to_um(WSI_Y_DS):.0f} µm")

    # Build tile grid
    tiles, n_cols, n_rows = build_tile_grid(WSI_X_DS, WSI_Y_DS, CORE_DS, HALO_DS)
    ts(f"Estimated runtime: ~{len(tiles) * 13 / 60:.1f}h "
       f"at ~13 min/tile ({len(tiles)} tiles)")

    # Load ALL transcripts — no region filter (whole slide)
    ts("Loading transcripts (whole slide)...")
    pts_df = pd.read_parquet(
        TRANSCRIPTS_PAR,
        columns=['x', 'y', 'z', 'feature_name', 'cell_id']
    )
    ts(f"Loaded {len(pts_df):,} transcripts")

    # Filter to gene transcripts with QV>=20
    if 'qv' in pts_df.columns:
        pts_df = pts_df[pts_df['qv'] >= 20].copy()
        ts(f"After QV>=20 filter: {len(pts_df):,}")
    if 'is_gene' in pts_df.columns:
        pts_df = pts_df[pts_df['is_gene'] == True].copy()
        ts(f"After is_gene filter: {len(pts_df):,}")

    # Load merged masks (memory-mapped — only slices loaded per tile)
    ts("Loading merged masks (mmap)...")
    merged = np.load(MERGED_NPY, mmap_mode='r')
    ts(f"Merged masks: {merged.shape}")

    # Process tiles
    results = []
    for i, tile in enumerate(tiles):
        out = run_proseg_tile(tile, merged, pts_df, i, len(tiles))
        if out:
            results.append(out)
        # Progress report every 10 tiles
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate    = elapsed / (i + 1)
            remain  = rate * (len(tiles) - i - 1)
            ts(f"Progress: {i+1}/{len(tiles)} tiles done | "
               f"elapsed={elapsed/3600:.1f}h | "
               f"remaining~{remain/3600:.1f}h")

    ts(f"\nCompleted {len(results)}/{len(tiles)} tiles successfully")

    # Assemble — filtering happens here
    ts("\n=== Assembling and filtering ===")
    assemble_tiles()

    elapsed = time.time() - t0
    ts(f"=== COMPLETE — {elapsed/3600:.2f}h ({elapsed:.1f}s) ===")
    ts(f"Output: {OUT_DIR}/merged_proseg_wsi.parquet")