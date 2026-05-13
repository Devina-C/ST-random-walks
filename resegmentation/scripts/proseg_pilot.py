#!/usr/bin/env python3
"""
proseg_pilot.py
===============
Pilot Proseg tiling run on the 2x2 patch region (patches 0, 1, 4, 5).

Uses the globally merged Cellpose pixel canvas (merged_masks.npy) as prior
segmentation and the raw Xenium transcripts as evidence. Proseg refines cell
boundaries tile by tile using overlapping halos, then applies the centroid
rule to avoid duplication.

Pilot region (DS coords):
  x: 0 → 12816  (2 patch widths)
  y: 0 → 9368   (2 patch heights)

Tiling (DS coords):
  Core size:  3000 DS px = 1275 um
  Halo:        300 DS px = 127.5 um
  Grid: ceil(12816/3000) x ceil(9368/3000) = 5 x 4 = 20 tiles

Coordinate systems:
  DS pixel:   merged_masks.npy space (full_res // 2)
  Full_res:   DS * 2
  Micron:     full_res * 0.2125  →  DS * 0.425

Proseg inputs:
  --cellpose-masks      : cropped merged_masks tile (npy.gz)
  --cellpose-scale      : 0.425 (DS pixel size in microns)
  --cellpose-x-transform: 0.425 0 <x_offset_um>
  --cellpose-y-transform: 0 0.425 <y_offset_um>
  transcripts CSV       : cropped to tile bounds in microns

Output:
  data/proseg_pilot/tile_RxC/    per-tile Proseg zarr
  data/proseg_pilot/shapes/      filtered shapes per tile (parquet)
  data/proseg_pilot/merged_proseg_pilot.parquet  stitched global shapes

Run on a compute node:
  sbatch scripts/slurm/submit_proseg_pilot.sh
  (requires --mem=32G --cpus-per-task=8 --time=04:00:00)
"""

import os, gc, gzip, shutil, subprocess, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

PROSEG_BIN      = "/users/k22026807/.cargo/bin/proseg"
TRANSCRIPTS_PAR = ("/scratch/users/k22026807/masters/project/xenium_output/"
                   "BC_prime.zarr/points/transcripts/points.parquet")
MERGED_NPY      = "data/merged/merged_masks.npy"
OUT_DIR         = "data/proseg_pilot"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/shapes", exist_ok=True)

# Coordinate system
PIXEL_SIZE_UM = 0.2125   # full_res µm/px
DS_FACTOR     = 2
DS_UM         = PIXEL_SIZE_UM * DS_FACTOR   # 0.425 µm per DS pixel

# Pilot region in DS coords (patches 0,1,4,5)
PILOT_X_DS = 12816
PILOT_Y_DS = 9368

# Tiling
CORE_DS  = 3000   # core size in DS px
HALO_DS  = 300    # halo in DS px

# Proseg parameters (matching original reseg.py settings)
PROSEG_SAMPLES        = "1000"  # enough iterations to converge at fine voxel size
PROSEG_BURNIN_SAMPLES = "100"   # burnin iterations at coarse resolution
PROSEG_VOXEL_SIZE     = "0.425" # fine voxel size = DS pixel size in µm
PROSEG_BURNIN_SIZE    = "0.850" # burnin voxel size = 2x fine (integer ratio required)
PROSEG_VOXEL_LAYERS   = "2"
PROSEG_NUCLEAR_PROB   = "0.25"
PROSEG_DIFF_PROB      = "0.25"
PROSEG_NTHREADS       = "8"

# Transcript filter threshold
MIN_TRANSCRIPTS       = 5       # cells with fewer transcripts are removed


# ── Timestamp helper ──────────────────────────────────────────────────────────

def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}", flush=True)


# ── Build tile grid ───────────────────────────────────────────────────────────

def build_tile_grid(pilot_x, pilot_y, core_size, halo_size):
    """
    Build list of (row, col, core_bounds, tile_bounds) in DS coords.
    core_bounds: (x0, y0, x1, y1) — strict property lines (half-open)
    tile_bounds: (x0, y0, x1, y1) — expanded by halo for Proseg
    """
    import math
    n_cols = math.ceil(pilot_x / core_size)
    n_rows = math.ceil(pilot_y / core_size)
    tiles  = []

    for row in range(n_rows):
        for col in range(n_cols):
            # Core bounds (half-open: min inclusive, max exclusive)
            cx0 = col * core_size
            cx1 = min((col + 1) * core_size, pilot_x)
            cy0 = row * core_size
            cy1 = min((row + 1) * core_size, pilot_y)

            # Tile bounds (core + halo, clamped to pilot region)
            tx0 = max(0,        cx0 - halo_size)
            tx1 = min(pilot_x,  cx1 + halo_size)
            ty0 = max(0,        cy0 - halo_size)
            ty1 = min(pilot_y,  cy1 + halo_size)

            tiles.append({
                'row': row, 'col': col,
                'core': (cx0, cy0, cx1, cy1),
                'tile': (tx0, ty0, tx1, ty1),
            })

    print(f"Tile grid: {n_cols} cols x {n_rows} rows = {len(tiles)} tiles")
    return tiles


# ── DS coords → microns ───────────────────────────────────────────────────────

def ds_to_um(ds_val):
    return ds_val * DS_UM


# ── Crop transcripts to tile ──────────────────────────────────────────────────

def crop_transcripts(pts_df, tx0_um, ty0_um, tx1_um, ty1_um):
    """
    Crop transcript dataframe to tile bounds in microns.
    Transcripts use (x=col direction, y=row direction) in µm.
    """
    mask = (
        (pts_df['x'] >= tx0_um) & (pts_df['x'] < tx1_um) &
        (pts_df['y'] >= ty0_um) & (pts_df['y'] < ty1_um)
    )
    return pts_df[mask].copy()


# ── Crop merged masks to tile ─────────────────────────────────────────────────

def crop_masks(merged, tx0, ty0, tx1, ty1):
    """
    Crop merged_masks.npy to tile bounds in DS coords.
    Returns cropped array with locally re-indexed IDs (preserves 0=background).
    """
    crop = merged[ty0:ty1, tx0:tx1].copy()

    # Re-index locally so IDs are compact (Proseg prefers this)
    uids = np.unique(crop); uids = uids[uids > 0]
    if len(uids) == 0:
        return crop
    remap        = np.zeros(int(crop.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(uids, start=1):
        remap[old_id] = new_id
    crop = remap[crop]
    return crop


# ── Save mask as gzipped npy ──────────────────────────────────────────────────

def save_npy_gz(arr, path):
    tmp = path.replace('.npy.gz', '.npy')
    np.save(tmp, arr)
    with open(tmp, 'rb') as f_in, gzip.open(path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(tmp)


# ── Run Proseg on one tile ────────────────────────────────────────────────────

def run_proseg_tile(tile, merged, pts_df, tile_idx):
    row, col   = tile['row'], tile['col']
    tx0, ty0, tx1, ty1 = tile['tile']    # DS coords
    cx0, cy0, cx1, cy1 = tile['core']    # DS coords

    ts(f"  Tile {tile_idx} (r{row}c{col}): "
       f"tile DS x[{tx0}:{tx1}] y[{ty0}:{ty1}], "
       f"core DS x[{cx0}:{cx1}] y[{cy0}:{cy1}]")

    # Convert tile bounds to microns for transcript cropping
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

    # ── Remap transcript cell_id to merged Cellpose IDs ───────────────────────
    # Transcripts have Xenium nuclear cell_id — replace with our merged mask ID
    # at each transcript position. Direct pixel lookup: convert micron coords
    # to local DS pixel coords within the cropped mask, index into mask array.
    tx_crop = tx_crop.copy()

    col_px = ((tx_crop["x"].values / DS_UM) - tx0).astype(int)
    row_px = ((tx_crop["y"].values / DS_UM) - ty0).astype(int)
    col_px = np.clip(col_px, 0, mask_crop.shape[1] - 1)
    row_px = np.clip(row_px, 0, mask_crop.shape[0] - 1)

    tx_crop["cell_id"] = mask_crop[row_px, col_px].astype(str)

    n_assigned   = (tx_crop["cell_id"] != "0").sum()
    n_unassigned = (tx_crop["cell_id"] == "0").sum()
    ts(f"    Assigned to merged cells: {n_assigned:,} | unassigned: {n_unassigned:,}")

    # Working directory for this tile
    tile_dir = os.path.join(OUT_DIR, f"tile_{row}x{col}")
    os.makedirs(tile_dir, exist_ok=True)

    tx_csv_path   = os.path.join(tile_dir, "transcripts.csv")
    mask_npy_path = os.path.join(tile_dir, "cellpose_masks.npy.gz")
    out_zarr_path = os.path.join(tile_dir, "proseg_out.zarr")

    # Save transcripts with remapped cell_id
    tx_out = tx_crop[["x", "y", "z", "feature_name", "cell_id"]].copy()
    tx_out.to_csv(tx_csv_path, index=False)

    # Save masks
    save_npy_gz(mask_crop.astype(np.uint32), mask_npy_path)
    del mask_crop; gc.collect()

    # Proseg affine transforms:
    # x_micron = DS_UM * x_pixel + x_offset_micron
    # y_micron = DS_UM * y_pixel + y_offset_micron
    x_offset_um = ds_to_um(tx0)
    y_offset_um = ds_to_um(ty0)

    cmd = [
        PROSEG_BIN, tx_csv_path,
        "-x", "x",
        "-y", "y",
        "-z", "z",
        "--gene-column",      "feature_name",
        "--cell-id-column",   "cell_id",
        "--cell-id-unassigned", "0",
        "--cellpose-masks",   mask_npy_path,
        "--cellpose-x-transform", str(DS_UM), "0", str(x_offset_um),
        "--cellpose-y-transform", "0", str(DS_UM), str(y_offset_um),
        "--samples",          PROSEG_SAMPLES,
        "--burnin-samples",   PROSEG_BURNIN_SAMPLES,
        "--voxel-size",       PROSEG_VOXEL_SIZE,
        "--burnin-voxel-size", PROSEG_BURNIN_SIZE,
        "--voxel-layers",     PROSEG_VOXEL_LAYERS,
        "--nuclear-reassignment-prob", PROSEG_NUCLEAR_PROB,
        "--diffusion-probability",     PROSEG_DIFF_PROB,
        "--nthreads",         PROSEG_NTHREADS,
        "--output-spatialdata", out_zarr_path,
        "--overwrite",
    ]

    ts(f"    Running Proseg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        ts(f"    ERROR: Proseg failed for tile {tile_idx}")
        print(result.stderr[-2000:])   # last 2000 chars of stderr
        return None

    ts(f"    Proseg complete")

    # Load refined shapes
    shapes_parquet = os.path.join(
        out_zarr_path, "shapes", "cell_boundaries", "shapes.parquet")
    if not os.path.exists(shapes_parquet):
        ts(f"    WARNING: no shapes parquet found")
        return None

    gdf = gpd.read_parquet(shapes_parquet)
    ts(f"    Raw shapes from Proseg: {len(gdf):,}")

    # ── Centroid rule: keep only cells in core region ─────────────────────────
    # Core bounds in microns (half-open)
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

    # Save tile shapes
    out_parquet = os.path.join(OUT_DIR, "shapes", f"tile_{row}x{col}.parquet")
    gdf_kept.to_parquet(out_parquet)

    # Cleanup tile inputs to save disk space
    os.remove(tx_csv_path)
    os.remove(mask_npy_path)

    del tx_crop, gdf, gdf_kept; gc.collect()
    return out_parquet


# ── Assemble all tile outputs ─────────────────────────────────────────────────

def assemble_tiles():
    """
    Concatenate filtered tile outputs and apply transcript count filter.

    Proseg saves per-cell transcript counts in the obs table of each tile zarr.
    We load these alongside the shapes and filter out any cell with fewer than
    MIN_TRANSCRIPTS assigned transcripts — removes background hallucinations
    and artefacts that Proseg kept but had no real transcript support.
    """
    import glob
    shape_files = sorted(glob.glob(os.path.join(OUT_DIR, "shapes", "tile_*.parquet")))
    if not shape_files:
        print("No tile outputs found to assemble")
        return

    ts(f"Assembling {len(shape_files)} tile outputs...")

    gdfs      = []
    for sf in shape_files:
        gdf_tile = gpd.read_parquet(sf)

        # Load transcript counts from tile zarr obs table
        # tile parquet name: shapes/tile_RxC.parquet → tile dir: tile_RxC/
        tile_name = os.path.basename(sf).replace(".parquet", "")
        obs_path  = os.path.join(OUT_DIR, tile_name,
                                 "proseg_out.zarr", "tables", "table", "obs")

        if os.path.exists(obs_path):
            try:
                import anndata as ad
                adata = ad.read_zarr(
                    os.path.join(OUT_DIR, tile_name, "proseg_out.zarr",
                                 "tables", "table"))
                counts = adata.obs[['transcript_count']].copy()                     if 'transcript_count' in adata.obs.columns                     else adata.obs[['cell_count']].copy()                     if 'cell_count' in adata.obs.columns                     else None

                if counts is not None:
                    col = counts.columns[0]
                    # Reset index on both for alignment
                    counts = counts.reset_index(drop=True)
                    gdf_tile = gdf_tile.reset_index(drop=True)
                    gdf_tile['transcript_count'] = counts[col].values
                    before = len(gdf_tile)
                    gdf_tile = gdf_tile[
                        gdf_tile['transcript_count'] >= MIN_TRANSCRIPTS].copy()
                    ts(f"  {tile_name}: {before} → {len(gdf_tile)} "
                       f"(removed {before-len(gdf_tile)} low-count cells)")
                else:
                    ts(f"  {tile_name}: no transcript_count column found "
                       f"— skipping count filter")
            except Exception as e:
                ts(f"  {tile_name}: could not load obs table ({e}) "
                   f"— skipping count filter")
        else:
            ts(f"  {tile_name}: obs path not found — skipping count filter")

        gdfs.append(gdf_tile)

    merged_gdf = pd.concat(gdfs, ignore_index=True)

    # Assign globally unique contiguous cell IDs
    merged_gdf['cell_id'] = [f"proseg_{i+1}" for i in range(len(merged_gdf))]

    ts(f"Total cells after transcript filter (>={MIN_TRANSCRIPTS}): "
       f"{len(merged_gdf):,}")

    out_path = os.path.join(OUT_DIR, "merged_proseg_pilot.parquet")
    merged_gdf.to_parquet(out_path)
    ts(f"Saved → {out_path}")
    return merged_gdf


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("=== PROSEG PILOT START ===")

    # Build tile grid
    tiles = build_tile_grid(PILOT_X_DS, PILOT_Y_DS, CORE_DS, HALO_DS)

    # Load transcripts (x,y columns only for cropping — load full later per tile)
    ts("Loading transcripts...")
    pts_df = pd.read_parquet(
        TRANSCRIPTS_PAR,
        columns=['x', 'y', 'z', 'feature_name', 'cell_id']
    )
    ts(f"Loaded {len(pts_df):,} transcripts")

    # Filter to pilot region only (save memory)
    pilot_x_um = ds_to_um(PILOT_X_DS)
    pilot_y_um = ds_to_um(PILOT_Y_DS)
    pts_df = pts_df[
        (pts_df['x'] < pilot_x_um) &
        (pts_df['y'] < pilot_y_um)
    ].copy()
    ts(f"Transcripts in pilot region: {len(pts_df):,}")

    # Load merged masks (memory-mapped — only load what we need per tile)
    ts("Loading merged masks (mmap)...")
    merged = np.load(MERGED_NPY, mmap_mode='r')
    ts(f"Merged masks: {merged.shape}")

    # Process tiles
    results = []
    for i, tile in enumerate(tiles):
        ts(f"\n--- Tile {i+1}/{len(tiles)} ---")
        out = run_proseg_tile(tile, merged, pts_df, i)
        if out:
            results.append(out)

    ts(f"\nCompleted {len(results)}/{len(tiles)} tiles successfully")

    # Assemble
    ts("\n=== Assembling ===")
    assemble_tiles()

    elapsed = time.time() - t0
    ts(f"=== COMPLETE — {elapsed/3600:.2f}h ({elapsed:.1f}s) ===")