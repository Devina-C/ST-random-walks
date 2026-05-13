#!/usr/bin/env python3
"""
merge_masks_full.py
===================
Full merge and deduplication of all 31 original patches with seam lattice masks.

Uses cell_labels_reseg (raw Cellpose pixel masks) for the pixel canvas and
deduplication. Shapes are derived from the merged pixel canvas using rasterio.

This is the original working approach that produced correct results.
Proseg refinement is deferred to transcript assignment stage.

Output:
  data/merged/merged_masks.npy          — global DS pixel label array (37473, 25633)
  data/merged/merged_cell_stats.csv     — cell_id, centroid, area per cell
  data/merged/merged_shapes.parquet     — polygon boundaries in DS coords
  data/merged/merged_global.zarr        — spatialdata zarr (Labels + Shapes)

Coordinate system: DS space (factor_rescale=2)
  1 DS pixel = 2 full_res pixels
  Label arrays are full_res → downsampled by FACTOR before placement

Run on a compute node:
  sbatch scripts/slurm/submit_merge.sh
  (requires --mem=64G --cpus-per-task=8 --time=06:00:00)
"""

import os, gc, warnings, time
from datetime import datetime
import numpy as np
import spatialdata as sd
from spatialdata.models import Labels2DModel, ShapesModel
from spatialdata import SpatialData
from skimage.measure import regionprops_table
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import from_bounds
import shapely.geometry
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import dask.array as da
warnings.filterwarnings("ignore")

# ── Timestamp helper ─────────────────────────────────────────────────────────

def ts(label=""):
    """Print current timestamp with optional label."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}")


# ── Config ────────────────────────────────────────────────────────────────────

WHOLE_ROI      = {'width': 51265, 'height': 74945}
N_COLS, N_ROWS = 4, 8
FACTOR         = 2
MARGIN         = 1500
MARGIN_DS      = MARGIN // FACTOR   # 750

IOU_THRESHOLD   = 0.3
MISSING_PATCHES = {28}

PATCHES_DIR  = "data/patches_core"
LATTICE_PATH = "data/seams_masks/lattice_masks.npy"
OUT_DIR      = "data/merged"
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_W = WHOLE_ROI['width']  / N_COLS   # 12816.25
PATCH_H = WHOLE_ROI['height'] / N_ROWS   # 9368.125
PW_DS   = int(PATCH_W // FACTOR)          # 6408
PH_DS   = int(PATCH_H // FACTOR)          # 4684

GLOBAL_H = int(WHOLE_ROI['height'] // FACTOR) + 1  # 37473
GLOBAL_W = int(WHOLE_ROI['width']  // FACTOR) + 1  # 25633

ALL_PATCHES = [i for i in range(N_COLS * N_ROWS) if i not in MISSING_PATCHES]


# ── Grid helpers ──────────────────────────────────────────────────────────────

def patch_offset_ds(patch_index):
    row = patch_index // N_COLS
    col = patch_index % N_COLS
    return int(row * PATCH_H // FACTOR), int(col * PATCH_W // FACTOR)


def patch_boundaries_ds(patch_index):
    row = patch_index // N_COLS
    col = patch_index % N_COLS
    y_off, x_off = patch_offset_ds(patch_index)
    return {
        'top':    y_off          if row > 0           else None,
        'bottom': y_off + PH_DS  if row < N_ROWS - 1 else None,
        'left':   x_off          if col > 0           else None,
        'right':  x_off + PW_DS  if col < N_COLS - 1 else None,
    }


def seam_strip_edges_ds(seam_name):
    if seam_name.startswith("seam_H_"):
        idx          = int(seam_name.split("_")[2])
        row_boundary = idx // N_COLS
        y_center     = (row_boundary + 1) * PATCH_H
        edge_a = max(0,        int((y_center - MARGIN) // FACTOR))
        edge_b = min(GLOBAL_H, int((y_center + MARGIN) // FACTOR))
        return edge_a, edge_b, 'y'
    else:
        v_idx        = int(seam_name.split("_")[2])
        col_boundary = v_idx % (N_COLS - 1)
        x_center     = (col_boundary + 1) * PATCH_W
        edge_a = max(0,        int((x_center - MARGIN) // FACTOR))
        edge_b = min(GLOBAL_W, int((x_center + MARGIN) // FACTOR))
        return edge_a, edge_b, 'x'


def seam_zone_ds(seam_name):
    off = np.load(f"data/seams_flows/{seam_name}_offsets.npy")
    dP  = np.load(f"data/seams_flows/{seam_name}_dP.npy")
    y0, x0 = int(off[0]), int(off[1])
    _, h, w = dP.shape
    del dP
    return y0, min(y0 + h, GLOBAL_H), x0, min(x0 + w, GLOBAL_W)


def all_seam_names():
    names = []
    for i in range((N_ROWS - 1) * N_COLS):
        names.append(f"seam_H_{i:02d}")
    for i in range(N_ROWS * (N_COLS - 1)):
        names.append(f"seam_V_{i:02d}")
    return names


# ── Phase 1: Build global patch canvas ───────────────────────────────────────

def build_patch_canvas():
    print("\n=== Phase 1: Building global patch canvas ===")
    canvas       = np.zeros((GLOBAL_H, GLOBAL_W), dtype=np.int32)
    patch_id_map = {}
    patch_bounds = {}
    id_offset    = 0

    for pi in tqdm(ALL_PATCHES, desc="Placing patches"):
        zarr_path = os.path.join(
            PATCHES_DIR,
            f"integrated_proseg_patch_{pi:02d}_BC_prime.zarr"
        )
        if not os.path.exists(zarr_path):
            print(f"  WARNING: patch {pi:02d} not found, skipping")
            continue

        sdata    = sd.read_zarr(zarr_path)
        arr_full = sdata.labels['cell_labels_reseg'].values.astype(np.int32)
        arr      = arr_full[::FACTOR, ::FACTOR]   # DS
        del arr_full, sdata

        y_off, x_off = patch_offset_ds(pi)
        h, w   = arr.shape
        y_max  = min(y_off + h, GLOBAL_H)
        x_max  = min(x_off + w, GLOBAL_W)
        h_clip = y_max - y_off
        w_clip = x_max - x_off
        arr_clip = arr[:h_clip, :w_clip]

        local_max = int(arr_clip.max())
        if local_max == 0:
            print(f"  Patch {pi:02d}: empty, skipping")
            del arr, arr_clip; gc.collect(); continue

        shifted = np.where(
            arr_clip > 0,
            arr_clip + id_offset,
            0
        ).astype(np.int32)

        canvas[y_off:y_max, x_off:x_max] = shifted

        for lid in range(1, local_max + 1):
            patch_id_map[lid + id_offset] = pi
        patch_bounds[pi] = patch_boundaries_ds(pi)
        id_offset += local_max

        del arr, arr_clip, shifted; gc.collect()

    print(f"Patch canvas: {id_offset} cells, "
          f"{(canvas > 0).sum()} foreground px")
    return canvas, patch_id_map, patch_bounds


# ── IoU and distance helpers ──────────────────────────────────────────────────

def compute_pairwise_iou(orig_crop, seam_crop):
    both = (orig_crop > 0) & (seam_crop > 0)
    if not both.any():
        return []
    ov = orig_crop[both]; sv = seam_crop[both]
    pairs, counts = np.unique(
        np.stack([ov, sv], axis=1), axis=0, return_counts=True)
    oa = {o: int((orig_crop == o).sum()) for o in np.unique(ov)}
    sa = {s: int((seam_crop == s).sum()) for s in np.unique(sv)}
    out = []
    for (o, s), inter in zip(pairs, counts):
        union = oa[o] + sa[s] - inter
        iou   = inter / union if union > 0 else 0.0
        if iou >= IOU_THRESHOLD:
            out.append((int(o), int(s), float(iou)))
    return out


def get_centroids(crop):
    """
    Compute centroids for all cells in a crop in one vectorised pass.
    Much faster than calling np.where per cell.
    Returns dict: cell_id → (cy, cx) in crop-local coords.
    """
    if crop.max() == 0:
        return {}
    props = regionprops_table(crop, properties=['label', 'centroid'])
    return {
        int(label): (float(cy), float(cx))
        for label, cy, cx in zip(
            props['label'],
            props['centroid-0'],
            props['centroid-1'],
        )
    }


def dist_to_nearest_patch_boundary(cy, cx, boundaries):
    dists = []
    if boundaries['top']    is not None: dists.append(abs(cy - boundaries['top']))
    if boundaries['bottom'] is not None: dists.append(abs(cy - boundaries['bottom']))
    if boundaries['left']   is not None: dists.append(abs(cx - boundaries['left']))
    if boundaries['right']  is not None: dists.append(abs(cx - boundaries['right']))
    return min(dists) if dists else float('inf')


def dist_to_seam_edge(cy, cx, edge_a, edge_b, axis):
    if axis == 'y': return min(abs(cy - edge_a), abs(cy - edge_b))
    else:           return min(abs(cx - edge_a), abs(cx - edge_b))


# ── Phase 2: Deduplication ────────────────────────────────────────────────────

def deduplicate(patch_canvas, patch_id_map, patch_bounds, lattice_masks):
    print("\n=== Phase 2: Deduplication ===")
    seam_canvas = lattice_masks.copy()
    stats = {'pairs': 0, 'patch_wins': 0, 'seam_wins': 0}

    for seam_name in tqdm(all_seam_names(), desc="Deduplicating"):
        off_path = f"data/seams_flows/{seam_name}_offsets.npy"
        if not os.path.exists(off_path): continue

        y0, y1, x0, x1      = seam_zone_ds(seam_name)
        edge_a, edge_b, axis = seam_strip_edges_ds(seam_name)

        orig_crop = patch_canvas[y0:y1, x0:x1].copy()
        seam_crop = seam_canvas[y0:y1,  x0:x1].copy()

        if orig_crop.max() == 0 or seam_crop.max() == 0:
            del orig_crop, seam_crop; continue

        pairs = compute_pairwise_iou(orig_crop, seam_crop)
        if not pairs:
            del orig_crop, seam_crop; continue

        stats['pairs'] += len(pairs)

        # Compute all centroids in one vectorised pass per crop
        orig_centroids = get_centroids(orig_crop)
        seam_centroids = get_centroids(seam_crop)

        orig_suppress = set()
        seam_suppress = set()

        for orig_id, seam_id, iou in pairs:
            o_local = orig_centroids.get(orig_id)
            s_local = seam_centroids.get(seam_id)
            if o_local is None or s_local is None:
                continue

            gcy_o = o_local[0] + y0;  gcx_o = o_local[1] + x0
            gcy_s = s_local[0] + y0;  gcx_s = s_local[1] + x0

            p_idx     = patch_id_map.get(orig_id)
            orig_dist = (dist_to_nearest_patch_boundary(
                             gcy_o, gcx_o, patch_bounds[p_idx])
                         if p_idx is not None else 0.0)
            seam_dist = dist_to_seam_edge(
                gcy_s, gcx_s, edge_a, edge_b, axis)

            if orig_dist >= seam_dist:
                seam_suppress.add(seam_id);  stats['patch_wins'] += 1
            else:
                orig_suppress.add(orig_id);  stats['seam_wins']  += 1

        if orig_suppress:
            m = np.isin(patch_canvas[y0:y1, x0:x1], list(orig_suppress))
            patch_canvas[y0:y1, x0:x1][m] = 0
        if seam_suppress:
            m = np.isin(seam_canvas[y0:y1, x0:x1], list(seam_suppress))
            seam_canvas[y0:y1, x0:x1][m] = 0

        del orig_crop, seam_crop, orig_centroids, seam_centroids
        gc.collect()

    print(f"Pairs evaluated: {stats['pairs']}")
    print(f"Patch wins:      {stats['patch_wins']}")
    print(f"Seam wins:       {stats['seam_wins']}")
    return seam_canvas


# ── Phase 3: Merge and re-index ───────────────────────────────────────────────

def merge_canvases(patch_canvas, seam_canvas):
    print("\n=== Phase 3: Merging ===")
    patch_max = int(patch_canvas.max())
    seam_nz   = seam_canvas > 0
    seam_canvas[seam_nz] += patch_max

    merged = patch_canvas.copy()
    fill   = (merged == 0) & seam_nz
    merged[fill] = seam_canvas[fill]

    print("Re-indexing to contiguous IDs...")
    uids  = np.unique(merged); uids = uids[uids > 0]
    remap = np.zeros(int(merged.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(uids, start=1):
        remap[old_id] = new_id
    merged = remap[merged]

    print(f"Total cells: {merged.max()}")
    return merged


# ── Phase 4: Derive shapes ────────────────────────────────────────────────────

def derive_shapes(merged, chunk_rows=2000):
    """
    Derive pixel-accurate polygon boundaries from merged pixel canvas.
    Uses rasterio.features.shapes — vectorised C-level, one pass per chunk.
    Much faster than find_contours per cell.
    Returns GeoDataFrame in DS coords.
    """
    print("\n=== Phase 4: Deriving polygon shapes ===")
    all_polys = []
    all_ids   = []
    n_chunks  = (GLOBAL_H + chunk_rows - 1) // chunk_rows

    for chunk_idx in tqdm(range(n_chunks), desc="Deriving shapes"):
        y0   = chunk_idx * chunk_rows
        y1   = min(y0 + chunk_rows, GLOBAL_H)
        crop = merged[y0:y1, :].astype(np.int32)

        transform = from_bounds(0, y0, GLOBAL_W, y1, GLOBAL_W, y1 - y0)

        for geom, val in rasterio_shapes(
                crop,
                mask=(crop > 0).astype(np.uint8),
                transform=transform):
            if val == 0:
                continue
            poly = shapely.geometry.shape(geom)
            if poly.is_empty:
                continue
            all_polys.append(poly)
            all_ids.append(int(val))

        del crop
        gc.collect()

    gdf = gpd.GeoDataFrame({'cell_id': all_ids}, geometry=all_polys, crs=None)
    print(f"Derived {len(gdf)} cell polygons")

    # ── Shape filter ──────────────────────────────────────────────────
    print("Applying shape quality filters...")
    DS_UM = 0.425
     # Convert to microns for area threshold
    gdf['area_um2']     = gdf.geometry.area * (DS_UM ** 2)
    gdf['convexity']    = gdf.geometry.area / gdf.geometry.convex_hull.area
    bounds              = gdf.geometry.bounds
    gdf['bbox_w']       = bounds['maxx'] - bounds['minx']
    gdf['bbox_h']       = bounds['maxy'] - bounds['miny']
    gdf['aspect_ratio'] = (gdf[['bbox_w','bbox_h']].max(axis=1) /
                           gdf[['bbox_w','bbox_h']].min(axis=1).clip(0.1))

    n_before = len(gdf)
    mask = (
        (gdf['area_um2']     >= 25.0)   &
        (gdf['area_um2']     <= 5000.0) &
        (gdf['convexity']    >= 0.7)    &
        (gdf['aspect_ratio'] <= 10.0)
    )
    gdf = gdf[mask].copy().reset_index(drop=True)
    print(f"  Removed {n_before - len(gdf):,} shapes "
          f"({(n_before-len(gdf))/n_before*100:.1f}%)")
    print(f"  Remaining: {len(gdf):,} shapes")
    
    # Drop helper columns — keep geometry and cell_id only
    gdf = gdf[['cell_id', 'geometry']].copy()

    return gdf


# ── Phase 5: Cell stats ───────────────────────────────────────────────────────

def compute_cell_stats(merged):
    print("\n=== Phase 5: Computing cell stats ===")
    props = regionprops_table(
        merged, properties=['label', 'area', 'centroid'])
    df = pd.DataFrame(props)
    df = df.rename(columns={'label': 'cell_id', 'area': 'area_px_ds'})

    if 'centroid-0' in df.columns:
        df = df.rename(columns={'centroid-0': 'centroid_y_ds',
                                 'centroid-1': 'centroid_x_ds'})
    elif 'centroid' in df.columns:
        df['centroid_y_ds'] = df['centroid'].apply(lambda v: v[0])
        df['centroid_x_ds'] = df['centroid'].apply(lambda v: v[1])
        df = df.drop(columns=['centroid'])

    df['centroid_y'] = df['centroid_y_ds'] * FACTOR
    df['centroid_x'] = df['centroid_x_ds'] * FACTOR
    df['area_px']    = df['area_px_ds'] * (FACTOR ** 2)

    print(f"Cell count: {len(df)}")
    print(f"Area (DS px) — median: {df['area_px_ds'].median():.0f}, "
          f"mean: {df['area_px_ds'].mean():.0f}")
    return df


# ── Phase 6: Save spatialdata zarr ───────────────────────────────────────────

def save_spatialdata_zarr(merged, gdf):
    print("\n=== Phase 6: Saving spatialdata zarr ===")
    zarr_path = os.path.join(OUT_DIR, "merged_global.zarr")

    merged_da   = da.from_array(merged, chunks=(4096, 4096))
    labels_elem = Labels2DModel.parse(
        merged_da,
        dims=('y', 'x'),
        scale_factors=[2, 2, 2, 2],
        transformations={'global': sd.transformations.Scale(
            [FACTOR, FACTOR], axes=('y', 'x'))},
    )
    shapes_elem = ShapesModel.parse(
        gdf,
        transformations={'global': sd.transformations.Scale(
            [FACTOR, FACTOR], axes=('y', 'x'))},
    )

    sdata = SpatialData(
        labels={'cell_labels_merged': labels_elem},
        shapes={'cell_boundaries_merged': shapes_elem},
    )
    sdata.write(zarr_path, overwrite=True)
    print(f"Saved: {zarr_path}")
    return zarr_path


# ── Phase 7: Build complete zarr (shapes + points + table) ───────────────────

def build_complete_zarr(merged, gdf, df_stats):
    print("\n=== Phase 7: Building complete zarr ===")
    import scipy.sparse as sp_sci
    import anndata as ad
    from spatialdata.models import PointsModel, TableModel
    from spatialdata.transformations import Identity

    XENIUM_ZARR = "/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr"
    TX_PARQUET  = f"{XENIUM_ZARR}/points/transcripts/points.parquet"
    OUT_COMPLETE = os.path.join(OUT_DIR, "merged_complete.zarr")

    DS_UM = 0.425  # DS pixel → micron

    # ── 7a: Filter shapes ─────────────────────────────────────────────────────
    t = time.time()
    ts("  7a: Filtering shapes...")

    gdf_um = gdf.copy()
    gdf_um['geometry'] = gdf_um.geometry.affine_transform(
        [DS_UM, 0, 0, DS_UM, 0, 0])
    gdf_um = gdf_um.set_crs(None, allow_override=True)

    gdf_um['area_um2']     = gdf_um.geometry.area
    gdf_um['convexity']    = gdf_um.geometry.area / \
                              gdf_um.geometry.convex_hull.area
    bounds                 = gdf_um.geometry.bounds
    gdf_um['bbox_w']       = bounds['maxx'] - bounds['minx']
    gdf_um['bbox_h']       = bounds['maxy'] - bounds['miny']
    gdf_um['aspect_ratio'] = gdf_um[['bbox_w','bbox_h']].max(axis=1) / \
                              gdf_um[['bbox_w','bbox_h']].min(axis=1).clip(0.1)

    MIN_AREA_UM2     = 25.0
    MAX_AREA_UM2     = 5000.0
    MIN_CONVEXITY    = 0.7
    MAX_ASPECT_RATIO = 10.0

    n_before = len(gdf_um)
    mask = (
        (gdf_um['area_um2']     >= MIN_AREA_UM2)     &
        (gdf_um['area_um2']     <= MAX_AREA_UM2)     &
        (gdf_um['convexity']    >= MIN_CONVEXITY)    &
        (gdf_um['aspect_ratio'] <= MAX_ASPECT_RATIO)
    )
    gdf_filt = gdf_um[mask].copy().reset_index(drop=True)
    ts(f"    {n_before:,} → {len(gdf_filt):,} shapes after QC "
       f"({n_before - len(gdf_filt):,} removed)")

    del gdf_um; gc.collect()
    ts(f"    Done in {time.time()-t:.1f}s")

    # ── 7b: Load transcripts ──────────────────────────────────────────────────
    t = time.time()
    ts("  7b: Loading transcripts...")
    tx = pd.read_parquet(
        TX_PARQUET,
        columns=['x', 'y', 'z', 'feature_name', 'qv',
                 'is_gene', 'overlaps_nucleus', 'transcript_id'])
    tx = tx[tx['is_gene'] == True].copy()
    tx = tx[tx['qv'] >= 20].copy()
    ts(f"    Gene transcripts (QV>=20): {len(tx):,}")

    # ── 7c: Spatial join ──────────────────────────────────────────────────────
    t = time.time()
    ts("  7c: Spatial join transcripts → cells (may take 20-40 min)...")
    tx_gdf = gpd.GeoDataFrame(
        tx, geometry=gpd.points_from_xy(tx['x'], tx['y']), crs=None)
    del tx; gc.collect()

    joined = gpd.sjoin(
        tx_gdf,
        gdf_filt[['cell_id', 'geometry']],
        how='left',
        predicate='within')
    assigned = joined[joined['cell_id'].notna()].copy()
    ts(f"    Assigned: {len(assigned):,} / {len(joined):,} transcripts "
       f"({len(assigned)/len(joined)*100:.1f}%)")
    del tx_gdf, joined; gc.collect()
    ts(f"    Done in {time.time()-t:.1f}s")

    # ── 7d: Keep all assigned cells  ─────────────────
    t = time.time()
    ts("  7d: Keeping all cells with >=1 transcript...")
    tx_counts = assigned.groupby('cell_id').size()
    valid_cells  = tx_counts[tx_counts >= 1].index
    gdf_final    = gdf_filt[gdf_filt['cell_id'].isin(valid_cells)].copy()
    gdf_final    = gdf_final.reset_index(drop=True)
    assigned_v   = assigned[assigned['cell_id'].isin(valid_cells)].copy()
    ts(f"    Cells with >=1 transcript: {len(gdf_final):,}")
    del gdf_filt; gc.collect()
    ts(f"    Done in {time.time()-t:.1f}s")

    # ── 7e: Build count matrix ────────────────────────────────────────────────
    t = time.time()
    ts("  7e: Building cell × gene count matrix...")
    genes    = sorted(assigned_v['feature_name'].unique())
    cell_ids = sorted(assigned_v['cell_id'].astype(int).unique().tolist())
    n_cells  = len(cell_ids)
    n_genes  = len(genes)
    ts(f"    {n_cells:,} cells × {n_genes:,} genes")

    cell_idx = {c: i for i, c in enumerate(cell_ids)}
    gene_idx = {g: i for i, g in enumerate(genes)}

    # drop rows where cell_id does not map
    assigned_v = assigned_v.copy()
    assigned_v['row_idx'] = assigned_v['cell_id'].astype(int).map(cell_idx)
    assigned_v = assigned_v[assigned_v['row_idx'].notna()].copy()
    assigned_v['col_idx'] = assigned_v['feature_name'].map(gene_idx)


    rows = assigned_v['row_idx'].astype(int).values
    cols = assigned_v['col_idx'].astype(int).values
    X    = sp_sci.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_genes))
    del rows, cols; gc.collect()
    ts(f"    Matrix shape: {X.shape}, nnz={X.nnz}")
    ts(f"    Done in {time.time()-t:.1f}s")

    # ── 7f: Build AnnData ─────────────────────────────────────────────────────
    t = time.time()
    ts("  7f: Building AnnData table...")

    # Align gdf_final to exactly the cell_ids in the count matrix
    gdf_final_aligned = gdf_final[
        gdf_final['cell_id'].astype(int).isin(cell_ids)].copy()
    gdf_final_aligned['cell_id_int'] = gdf_final_aligned['cell_id'].astype(int)
    gdf_final_aligned = gdf_final_aligned.set_index('cell_id_int').loc[cell_ids]
    gdf_final_aligned = gdf_final_aligned.reset_index(drop=True)

    ts(f"  Aligned shapes: {len(gdf_final_aligned):,} (matrix: {n_cells:,})")
    assert len(gdf_final_aligned) == n_cells, \
        f"Shape mismatch: {len(gdf_final_aligned)} shapes vs {n_cells} matrix rows"


    obs = pd.DataFrame({
        'cell_id':        [str(c) for c in cell_ids],
        'n_transcripts':  np.array(X.sum(axis=1)).flatten().astype(int),
        'area_um2':       gdf_final_aligned['area_um2'].values
                          if 'area_um2' in gdf_final_aligned.columns
                          else np.nan,
        'convexity':      gdf_final_aligned['convexity'].values
                          if 'convexity' in gdf_final_aligned.columns
                          else np.nan,
        'aspect_ratio':   gdf_final_aligned['aspect_ratio'].values
                          if 'aspect_ratio' in gdf_final_aligned.columns
                          else np.nan,
        'region':         'cell_boundaries_filtered',
    }, index=[str(c) for c in cell_ids])
    obs['region'] = obs['region'].astype('category')

    var   = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = [str(c) for c in cell_ids]
    adata.var_names = genes
    ts(f"    AnnData: {adata.shape}")
    ts(f"    n_transcripts: min={obs['n_transcripts'].min()}, "
       f"median={obs['n_transcripts'].median():.0f}, "
       f"max={obs['n_transcripts'].max()}")
    del X; gc.collect()
    ts(f"    Done in {time.time()-t:.1f}s")

    # ── 7g: Prepare spatialdata elements ──────────────────────────────────────
    t = time.time()
    ts("  7g: Preparing spatialdata elements...")

    # Shapes — filtered, in microns
    shapes_sd = ShapesModel.parse(
        gdf_final_aligned[['geometry']].copy(),
        transformations={'global': Identity()})

    # Points — all QV>=20 gene transcripts
    tx_pts = pd.read_parquet(
        TX_PARQUET,
        columns=['x', 'y', 'z', 'feature_name', 'qv',
                 'is_gene', 'transcript_id'])
    tx_pts = tx_pts[tx_pts['is_gene'] == True].copy()
    tx_pts = tx_pts[tx_pts['qv'] >= 20].copy()
    tx_pts = tx_pts.rename(columns={'feature_name': 'gene'})
    tx_pts['transcript_id'] = tx_pts['transcript_id'].astype(str)
    points_sd = PointsModel.parse(
        tx_pts,
        coordinates={'x': 'x', 'y': 'y', 'z': 'z'},
        feature_key='gene',
        instance_key='transcript_id',
        transformations={'global': Identity()})
    del tx_pts; gc.collect()

    # Table
    table_sd = TableModel.parse(
        adata,
        region='cell_boundaries_filtered',
        region_key='region',
        instance_key='cell_id')

    # Labels — reuse existing merged labels
    ts("  Loading existing labels from merged zarr...")
    existing = sd.read_zarr(os.path.join(OUT_DIR, "merged_global.zarr"))
    labels_elem = existing.labels['cell_labels_merged']
    ts(f"    Done in {time.time()-t:.1f}s")

    # ── 7h: Write complete zarr ───────────────────────────────────────────────
    t = time.time()
    ts(f"  7h: Writing complete zarr to {OUT_COMPLETE}...")
    sdata_complete = sd.SpatialData(
        labels={'cell_labels_merged':       labels_elem},
        shapes={
            'cell_boundaries_merged':   existing.shapes['cell_boundaries_merged'],
            'cell_boundaries_filtered': shapes_sd,
        },
        points={'transcripts': points_sd},
        tables={'table':       table_sd},
    )
    if os.path.exists(OUT_COMPLETE):
        import shutil
        shutil.rmtree(OUT_COMPLETE)
    sdata_complete.write(OUT_COMPLETE)
    ts(f"  Saved: {OUT_COMPLETE}")

    ts("  Summary:")
    ts(f"    Input shapes:    850,601")
    ts(f"    After QC:        {len(gdf_final):,}")
    ts(f"    Cells (>=1 tx):  {n_cells:,}")
    ts(f"    Genes:           {n_genes}")
    ts(f"    Transcripts:     {len(assigned):,} assigned")
    ts(f"    Done in {time.time()-t:.1f}s")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("START")

    # Phase 1
    ts("Phase 1 start — building patch canvas")
    patch_canvas, patch_id_map, patch_bounds = build_patch_canvas()
    ts("Phase 1 complete")

    # Load seam lattice masks
    ts("Loading lattice masks")
    lattice = np.load(LATTICE_PATH)
    assert lattice.shape == (GLOBAL_H, GLOBAL_W), \
        f"Unexpected lattice shape: {lattice.shape}"
    print(f"Lattice: {lattice.max()} seam cells")
    ts("Lattice loaded")

    # Phase 2
    ts("Phase 2 start — deduplication")
    seam_canvas = deduplicate(
        patch_canvas, patch_id_map, patch_bounds, lattice)
    del lattice; gc.collect()
    ts("Phase 2 complete")

    # Phase 3
    ts("Phase 3 start — merging")
    merged = merge_canvases(patch_canvas, seam_canvas)
    del patch_canvas, seam_canvas; gc.collect()
    ts("Phase 3 complete")

    # Save pixel canvas
    out_npy = os.path.join(OUT_DIR, "merged_masks.npy")
    np.save(out_npy, merged)
    print(f"Saved: {out_npy}")
    ts("merged_masks.npy saved")

    # Phase 4
    ts("Phase 4 start — deriving shapes")
    gdf = derive_shapes(merged)
    gdf.to_parquet(os.path.join(OUT_DIR, "merged_shapes.parquet"))
    print(f"Saved: merged_shapes.parquet")
    ts("Phase 4 complete")

    # Phase 5
    ts("Phase 5 start — cell stats")
    df_stats = compute_cell_stats(merged)
    df_stats.to_csv(os.path.join(OUT_DIR, "merged_cell_stats.csv"), index=False)
    print(f"Saved: merged_cell_stats.csv")
    ts("Phase 5 complete")

    # Phase 6
    ts("Phase 6 start — saving spatialdata zarr")
    save_spatialdata_zarr(merged, gdf)
    ts("Phase 6 complete")

    # Phase 7
    ts("Phase 7 start - building complete zarr")
    build_complete_zarr(merged, gdf, df_stats)
    ts("Phase 7 complete")

    elapsed = time.time() - t0
    ts(f"COMPLETE — total time {elapsed/3600:.2f}h ({elapsed:.1f}s)")