#!/usr/bin/env python3
"""
viz_wholeslide.py
=================
Produces a single high-resolution whole-slide PNG showing merged cell
boundaries overlaid on the DAPI image.

The output image is rendered at 1 DS pixel = 1 image pixel, so the full
output is (37473 x 25633) pixels. Saved as PNG — open in any image viewer
and zoom into any boundary region you want to inspect.

Two outputs:
  data/merged/viz/wholeslide_pixels.png    — random-colour filled cell masks
  data/merged/viz/wholeslide_polygons.png  — cell outlines on DAPI background

Run on a compute node:
  sbatch scripts/slurm/submit_wholeslide_viz.sh
  (requires --mem=48G --time=04:00:00)
"""

import os, gc, warnings
import numpy as np
import spatialdata as sd
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import from_bounds
import shapely.geometry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────

XENIUM_ZARR = "/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr"
MERGED_NPY  = "data/merged/merged_masks.npy"
OUT_DIR     = "data/merged/viz"
os.makedirs(OUT_DIR, exist_ok=True)

WHOLE_ROI    = {'width': 51265, 'height': 74945}
N_COLS, N_ROWS = 4, 8
FACTOR       = 2

PATCH_W = WHOLE_ROI['width']  / N_COLS
PATCH_H = WHOLE_ROI['height'] / N_ROWS
PW_DS   = int(PATCH_W // FACTOR)   # 6408
PH_DS   = int(PATCH_H // FACTOR)   # 4684

GLOBAL_H = int(WHOLE_ROI['height'] // FACTOR) + 1  # 37473
GLOBAL_W = int(WHOLE_ROI['width']  // FACTOR) + 1  # 25633


# ── Load DAPI at DS resolution ────────────────────────────────────────────────

def load_dapi_fullslide():
    """
    Load full-slide DAPI at DS resolution (factor=2 from full_res).
    Uses pyramid level 1 (2x from full_res = DS resolution).
    Falls back to level 2 if level 1 is too large.
    Returns array (GLOBAL_H, GLOBAL_W) normalised to [0,1].
    """
    print("Loading full-slide DAPI...")
    sdata  = sd.read_zarr(XENIUM_ZARR)
    img_dt = sdata.images[list(sdata.images.keys())[0]]

    for lv in range(1, 5):
        ps = 2 ** lv
        try:
            ds  = img_dt[f'scale{lv}'].ds
            var = list(ds.data_vars)[0]
            arr = ds[var]
            print(f"  Trying level {lv} (scale={ps}x): shape={arr.shape}")

            if arr.ndim == 3:
                img = arr[0].values
            else:
                img = arr.values

            img = img.astype(np.float32)
            p1, p99 = np.percentile(img[img > 0], [1, 99]) if (img > 0).any() else (0, 1)
            img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)

            print(f"  Loaded: {img.shape}")
            del sdata
            return img, ps

        except Exception as e:
            print(f"  Level {lv} failed: {e}")
            continue

    del sdata
    raise RuntimeError("Could not load DAPI")


# ── Pixel fill visualisation ──────────────────────────────────────────────────

def make_pixel_viz(merged):
    print("\n=== Pixel fill visualisation ===")
    print("  Building colour map...")
    rng   = np.random.default_rng(42)
    n_ids = int(merged.max()) + 1
    cmap  = np.zeros((n_ids, 3), dtype=np.uint8)
    cmap[1:] = rng.integers(40, 220, size=(n_ids - 1, 3))

    print(f"  Rendering ({GLOBAL_H} x {GLOBAL_W})...")
    rgb = np.zeros((GLOBAL_H, GLOBAL_W, 3), dtype=np.uint8)
    chunk_rows = 2000
    for ci in range((GLOBAL_H + chunk_rows - 1) // chunk_rows):
        y0 = ci * chunk_rows
        y1 = min(y0 + chunk_rows, GLOBAL_H)
        rgb[y0:y1] = cmap[merged[y0:y1, :]]

    # Patch grid lines
    for row in range(N_ROWS + 1):
        y = int(row * PATCH_H // FACTOR)
        if 0 <= y < GLOBAL_H:
            rgb[max(0,y-1):y+2, :] = [255, 255, 255]
    for col in range(N_COLS + 1):
        x = int(col * PATCH_W // FACTOR)
        if 0 <= x < GLOBAL_W:
            rgb[:, max(0,x-1):x+2] = [255, 255, 255]

    # Save directly with PIL — no DPI scaling, 1 array pixel = 1 image pixel
    from PIL import Image
    out_path = os.path.join(OUT_DIR, "wholeslide_pixels.png")
    print(f"  Saving {out_path}...")
    Image.fromarray(rgb).save(out_path)
    print(f"  Saved: {out_path}  ({GLOBAL_W}x{GLOBAL_H} px)")
    del rgb, cmap; gc.collect()


# ── Polygon outline visualisation ─────────────────────────────────────────────

def make_polygon_viz(merged, dapi, dapi_scale):
    """
    Cell boundary outlines overlaid on DAPI background.
    Uses rasterio to derive boundaries efficiently.
    DAPI may be at a different scale than DS — handles the rescaling.
    """
    print("\n=== Polygon outline visualisation ===")

    # dapi is at full_res / dapi_scale
    # merged is at DS = full_res / FACTOR
    # To overlay: resize dapi to DS resolution or vice versa
    # Simpler: work at dapi's native resolution, scale merged coords

    dapi_h, dapi_w = dapi.shape
    print(f"  DAPI shape: {dapi_h} x {dapi_w} (scale={dapi_scale}x from full_res)")
    print(f"  Merged shape: {GLOBAL_H} x {GLOBAL_W} (scale={FACTOR}x from full_res)")

    # Scale factor from DS coords to dapi pixel coords
    ds_to_dapi = FACTOR / dapi_scale   # e.g. if dapi_scale=2, ds_to_dapi=1.0

    # Build boundary image at dapi resolution using find_boundaries on chunks
    print("  Deriving cell boundaries...")
    boundary_img = np.zeros((dapi_h, dapi_w), dtype=np.uint8)

    chunk_rows_ds = 1000
    n_chunks      = (GLOBAL_H + chunk_rows_ds - 1) // chunk_rows_ds

    for ci in range(n_chunks):
        y0_ds = ci * chunk_rows_ds
        y1_ds = min(y0_ds + chunk_rows_ds, GLOBAL_H)

        # Convert DS row range to dapi pixel range
        y0_dp = int(y0_ds * ds_to_dapi)
        y1_dp = min(int(y1_ds * ds_to_dapi) + 1, dapi_h)

        if y0_dp >= dapi_h: break

        chunk = merged[y0_ds:y1_ds, :]

        # Downsample/upsample chunk to dapi resolution
        if ds_to_dapi != 1.0:
            from skimage.transform import resize
            chunk_dapi = resize(
                chunk.astype(np.float32),
                (y1_dp - y0_dp, dapi_w),
                order=0,   # nearest neighbour — preserve label IDs
                anti_aliasing=False
            ).astype(np.int32)
        else:
            h_clip = min(chunk.shape[0], y1_dp - y0_dp)
            chunk_dapi = chunk[:h_clip, :dapi_w]

        # Find boundaries
        from skimage.segmentation import find_boundaries
        bnd = find_boundaries(chunk_dapi, mode='inner')
        boundary_img[y0_dp:y0_dp + bnd.shape[0], :bnd.shape[1]] |= bnd.astype(np.uint8)

        del chunk, chunk_dapi, bnd
        gc.collect()

    # Composite: DAPI as grey background, boundaries as white
    print("  Compositing...")
    rgb = np.stack([dapi, dapi, dapi], axis=2)   # grey DAPI
    rgb[boundary_img > 0] = [1.0, 1.0, 1.0]      # white boundaries

    # Draw patch grid lines in red
    for row in range(N_ROWS + 1):
        y_ds = int(row * PATCH_H // FACTOR)
        y_dp = int(y_ds * ds_to_dapi)
        if 0 <= y_dp < dapi_h:
            rgb[max(0, y_dp-1):y_dp+2, :] = [1.0, 0.0, 0.0]

    for col in range(N_COLS + 1):
        x_ds = int(col * PATCH_W // FACTOR)
        x_dp = int(x_ds * ds_to_dapi)
        if 0 <= x_dp < dapi_w:
            rgb[:, max(0, x_dp-1):x_dp+2] = [1.0, 0.0, 0.0]

    rgb_uint8 = (rgb * 255).astype(np.uint8)

    out_path  = os.path.join(OUT_DIR, "wholeslide_polygons.png")
    print(f"  Saving {out_path}...")
    Image.fromarray(rgb_uint8).save(out_path)
    print(f"  Saved: {out_path}")

    del rgb, boundary_img, dapi
    gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("Loading merged masks...")
    merged = np.load(MERGED_NPY)
    print(f"Merged: shape={merged.shape}, cells={merged.max()}")

    # Pixel fill viz — uses merged only, no DAPI needed
    make_pixel_viz(merged)

    # Polygon viz — needs DAPI
    dapi, dapi_scale = load_dapi_fullslide()
    make_polygon_viz(merged, dapi, dapi_scale)

    print(f"\n=== COMPLETE in {time.time()-t0:.1f}s ===")
    print(f"Outputs:")
    print(f"  {OUT_DIR}/wholeslide_pixels.png")
    print(f"  {OUT_DIR}/wholeslide_polygons.png")