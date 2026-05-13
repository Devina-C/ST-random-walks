#!/usr/bin/env python3
"""
derive_shapes_only.py
=====================
Runs phases 4-6 only, loading from the already-saved merged_masks.npy.

Applies two filters before deriving shapes:

  1. Max area filter — removes large Cellpose artefacts (spider/flower shapes)
     Cells above MAX_AREA_DS are zeroed out. These are flow convergence
     artefacts where Cellpose assigned a huge irregular region to one cell.

  2. Tissue mask filter — removes background hallucinations
     Loads DAPI at DS resolution, thresholds to find tissue, dilates by
     TISSUE_DILATION_PX to preserve cells at tissue edges. Cells whose
     centroid falls outside the tissue mask are zeroed out. These are cells
     Cellpose detected in camera noise/background outside the tissue boundary.

Both filters operate on the pixel canvas before shape derivation, so the
output shapes, stats, and zarr all reflect the filtered result.

Requires rasterio: conda install -c conda-forge rasterio

Run on a compute node:
  sbatch --job-name=derive_shapes --partition=cpu --mem=48G
         --cpus-per-task=4 --time=01:00:00
         --output=logs/derive_shapes_%j.out
         --error=logs/derive_shapes_%j.err
         --wrap="cd /path/to/resegmentation &&
                 conda run -n xenium python -u scripts/derive_shapes.py"
"""

import os, gc, warnings, time
from datetime import datetime
import numpy as np
import spatialdata as sd
from spatialdata.models import Labels2DModel, ShapesModel
from spatialdata import SpatialData
from skimage.measure import regionprops_table
from scipy.ndimage import binary_dilation
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import dask.array as da
from rasterio.features import shapes
from rasterio.transform import from_bounds
import shapely.geometry
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

XENIUM_ZARR = "/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr"
MERGED_NPY  = "data/merged/merged_masks.npy"
OUT_DIR     = "data/merged"
os.makedirs(OUT_DIR, exist_ok=True)

FACTOR   = 2
GLOBAL_H = 37473
GLOBAL_W = 25633

# Filter thresholds
MAX_AREA_DS        = 5000   # DS px² — removes 132 spider/flower artefacts
TISSUE_DILATION_PX = 20     # DS px buffer around tissue edge


# ── Timestamp helper ──────────────────────────────────────────────────────────

def ts(label=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}")


# ── Filter 1: Max area ────────────────────────────────────────────────────────

def filter_max_area(merged, max_area_ds=MAX_AREA_DS):
    """
    Zero out cells whose area exceeds max_area_ds DS pixels².
    These are Cellpose flow convergence artefacts — large irregular
    spider/flower-shaped regions with no biological basis.
    """
    ts(f"Filter 1: max area (>{max_area_ds} DS px²)")

    props = regionprops_table(merged, properties=['label', 'area'])
    df    = pd.DataFrame(props)
    large = df[df['area'] > max_area_ds]['label'].values

    print(f"  Cells above threshold: {len(large)}")
    if len(large) == 0:
        return merged

    mask = np.isin(merged, large)
    merged[mask] = 0
    print(f"  Zeroed out {mask.sum()} pixels")
    return merged


# ── Filter 2: Tissue mask ─────────────────────────────────────────────────────

def load_tissue_mask():
    """
    Load DAPI at DS resolution (pyramid level 1 = scale 2x from full_res).
    Threshold at 2nd percentile of non-zero pixels to get tissue mask.
    Dilate by TISSUE_DILATION_PX to preserve cells at tissue edges.
    Returns boolean array (GLOBAL_H, GLOBAL_W).
    """
    ts("Loading DAPI for tissue mask...")
    sdata  = sd.read_zarr(XENIUM_ZARR)
    img_dt = sdata.images[list(sdata.images.keys())[0]]

    for lv in range(1, 5):
        ps = 2 ** lv
        try:
            ds_  = img_dt[f'scale{lv}'].ds
            var  = list(ds_.data_vars)[0]
            arr  = ds_[var]
            print(f"  Trying level {lv} (scale={ps}x): shape={arr.shape}")

            if arr.ndim == 3:
                img = arr[0].values
            else:
                img = arr.values

            img = img.astype(np.float32)
            del sdata, arr, ds_
            break
        except Exception as e:
            print(f"  Level {lv} failed: {e}")
            continue

    # Threshold: 2nd percentile of non-zero pixels
    nonzero = img[img > 0]
    thresh  = np.percentile(nonzero, 2)
    tissue  = img > thresh
    print(f"  DAPI shape: {img.shape}, threshold: {thresh:.2f}")
    print(f"  Raw tissue coverage: {tissue.mean()*100:.1f}%")

    # Dilate to include cells at tissue boundary
    tissue = binary_dilation(tissue, iterations=TISSUE_DILATION_PX)
    print(f"  Dilated tissue coverage: {tissue.mean()*100:.1f}%")

    del img, nonzero
    gc.collect()
    return tissue


def filter_tissue_mask(merged, tissue_mask):
    """
    Zero out cells whose centroid falls outside the tissue mask.
    These are background hallucinations — Cellpose detected apparent
    cells in camera noise/empty slide regions with no tissue.
    """
    ts("Filter 2: tissue mask")

    # Get centroids for all remaining cells
    props = regionprops_table(merged, properties=['label', 'centroid'])
    df    = pd.DataFrame(props)
    df    = df.rename(columns={'centroid-0': 'cy', 'centroid-1': 'cx'})

    cy = df['cy'].values.astype(int).clip(0, tissue_mask.shape[0] - 1)
    cx = df['cx'].values.astype(int).clip(0, tissue_mask.shape[1] - 1)

    in_tissue = tissue_mask[cy, cx]
    df['in_tissue'] = in_tissue

    outside_ids = df[~in_tissue]['label'].values
    print(f"  Cells in tissue:      {in_tissue.sum():,}")
    print(f"  Cells outside tissue: {len(outside_ids):,}")

    if len(outside_ids) == 0:
        return merged

    mask = np.isin(merged, outside_ids)
    merged[mask] = 0
    print(f"  Zeroed out {mask.sum()} pixels")
    return merged


# ── Re-index after filtering ──────────────────────────────────────────────────

def reindex(merged):
    """Re-index remaining cells to contiguous IDs 1..N after filtering."""
    ts("Re-indexing after filtering")
    uids  = np.unique(merged); uids = uids[uids > 0]
    remap = np.zeros(int(merged.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(uids, start=1):
        remap[old_id] = new_id
    merged = remap[merged]
    print(f"  Cells remaining: {merged.max():,}")
    return merged


# ── Phase 4: Derive shapes ────────────────────────────────────────────────────

def derive_shapes(merged, chunk_rows=2000):
    print("\n=== Phase 4: Deriving polygon shapes ===")
    all_polys = []
    all_ids   = []
    n_chunks  = (GLOBAL_H + chunk_rows - 1) // chunk_rows

    for chunk_idx in tqdm(range(n_chunks), desc="Deriving shapes"):
        y0   = chunk_idx * chunk_rows
        y1   = min(y0 + chunk_rows, GLOBAL_H)
        crop = merged[y0:y1, :].astype(np.int32)

        # positional args — compatible with all rasterio versions
        transform = from_bounds(0, y0, GLOBAL_W, y1, GLOBAL_W, y1 - y0)

        for geom, val in shapes(crop,
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

    print(f"Cell count: {len(df):,}")
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ts("START")

    ts("Loading merged_masks.npy")
    merged = np.load(MERGED_NPY)
    print(f"Loaded: shape={merged.shape}, cells={merged.max():,}")
    ts("Loaded")

    # ── Filtering ─────────────────────────────────────────────────────────────
    print(f"\n=== Filtering ===")
    print(f"Before filtering: {(merged > 0).sum():,} foreground px, "
          f"{merged.max():,} cells")

    # Filter 1: remove large artefacts
    merged = filter_max_area(merged, MAX_AREA_DS)

    # Filter 2: remove background hallucinations
    tissue_mask = load_tissue_mask()
    merged = filter_tissue_mask(merged, tissue_mask)
    del tissue_mask; gc.collect()

    # Re-index to contiguous IDs
    merged = reindex(merged)

    print(f"\nAfter filtering: {(merged > 0).sum():,} foreground px, "
          f"{merged.max():,} cells")

    # Save filtered pixel canvas
    filtered_npy = os.path.join(OUT_DIR, "merged_masks_filtered.npy")
    np.save(filtered_npy, merged)
    ts(f"Saved filtered canvas: {filtered_npy}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    ts("Phase 4 start — deriving shapes")
    gdf = derive_shapes(merged)
    gdf.to_parquet(os.path.join(OUT_DIR, "merged_shapes.parquet"))
    print("Saved: merged_shapes.parquet")
    ts("Phase 4 complete")

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    ts("Phase 5 start — cell stats")
    df_stats = compute_cell_stats(merged)
    df_stats.to_csv(os.path.join(OUT_DIR, "merged_cell_stats.csv"), index=False)
    print("Saved: merged_cell_stats.csv")
    ts("Phase 5 complete")

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    ts("Phase 6 start — saving spatialdata zarr")
    save_spatialdata_zarr(merged, gdf)
    ts("Phase 6 complete")

    elapsed = time.time() - t0
    ts(f"COMPLETE — total time {elapsed/3600:.2f}h ({elapsed:.1f}s)")