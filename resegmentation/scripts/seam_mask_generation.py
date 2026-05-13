# Phase 1: combine seam flows onto global canvas
# Phase 2: run compute_masks per-seam on local crops
# Phase 3: place results onto global mask canvas

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from cellpose import dynamics
import torch

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

if __name__ == "__main__":

    print("=== SEAM MASK GENERATION ===")

    seam_flow_dir = "data/seams_flows"
    out_dir       = "data/seams_masks"
    os.makedirs(out_dir, exist_ok=True)

    WHOLE_ROI      = {'width': 51265, 'height': 74945}
    factor_rescale = 2

    global_h = int(WHOLE_ROI['height'] // factor_rescale) + 1  # 37473
    global_w = int(WHOLE_ROI['width']  // factor_rescale) + 1  # 25633

    # ── PHASE 1: Combine seam flows onto global canvas ────────────────────────
    print(f"\nPhase 1: Allocating global canvas ({global_h} x {global_w})...")
    global_dP       = np.zeros((2, global_h, global_w), dtype=np.float32)
    global_cellprob = np.zeros((global_h, global_w),    dtype=np.float32)
    global_weights  = np.zeros((global_h, global_w),    dtype=np.float32)

    offset_files = sorted(glob.glob(os.path.join(seam_flow_dir, "*_offsets.npy")))
    print(f"Found {len(offset_files)} seam flow files")
    if len(offset_files) != 52:
        print(f"WARNING: expected 52, got {len(offset_files)}")

    tile_dist_cache = {}

    for offset_file in tqdm(offset_files, desc="Combining flows"):
        base     = offset_file.replace("_offsets.npy", "")
        offsets  = np.load(offset_file)
        dP       = np.load(f"{base}_dP.npy")
        cellprob = np.load(f"{base}_cellprob.npy")

        min_y, min_x = int(offsets[0]), int(offsets[1])
        _, h, w = dP.shape
        max_y = min(min_y + h, global_h)
        max_x = min(min_x + w, global_w)
        h_clip = max_y - min_y
        w_clip = max_x - min_x

        if (h, w) not in tile_dist_cache:
            padded = np.pad(np.ones((h, w), dtype=np.uint8), 1, mode='constant')
            d = distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)
            d /= d.max()
            tile_dist_cache[(h, w)] = d
        dist_map = tile_dist_cache[(h, w)][:h_clip, :w_clip]

        dP_weighted = dP[:, :h_clip, :w_clip] * dist_map
        global_dP[:, min_y:max_y, min_x:max_x]    += dP_weighted
        global_cellprob[min_y:max_y, min_x:max_x] += cellprob[:h_clip, :w_clip] * dist_map
        global_weights[min_y:max_y, min_x:max_x]  += dist_map

        del dP, cellprob, dP_weighted

    print("Normalising...")
    valid = global_weights > 0
    for c in range(2):
        global_dP[c][valid] /= global_weights[valid]
    global_cellprob[valid]  /= global_weights[valid]
    global_cellprob[~valid]  = -10.0
    del global_weights

    # ── PHASE 2: Global compute_masks on GPU ──────────────────────────────────
    print("\nPhase 2: Running compute_masks on global canvas...")

    device = torch.device('cpu')
    print(f"Using device: {device}")

    result = dynamics.compute_masks(
        global_dP,
        global_cellprob,
        niter=200,
        cellprob_threshold=-3.0,
        flow_threshold=1.0,
        do_3D=False,
        device=device,
    )
    lattice_masks = np.array(result)

    print(f"Mask shape: {lattice_masks.shape}")
    print(f"Total cells: {lattice_masks.max()}")

    # ── PHASE 3: Save ─────────────────────────────────────────────────────────
    print("\nPhase 3: Saving...")
    out_dir  = "data/seams_masks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lattice_masks.npy")
    np.save(out_path, lattice_masks)
    print(f"Saved: {out_path}")
    print("=== COMPLETE ===")