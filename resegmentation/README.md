# Resegmentation

Exploratory whole-slide cell segmentation pipeline using Cellpose and Proseg, developed in parallel to the main analysis pipeline to assess whether transcript-informed resegmentation could improve on the default Xenium output.

**Note**: this workflow was developed independently. Its outputs were not used in the downstream cell typing, graph construction or communication analyses described in the main pipeline (`05_random_walks/`), which are based on the default Xenium segmentation (699,110 cells).

## Pipeline (`scripts/`)

The whole-slide image (51,265 × 74,945 px) is processed as a 4×8 grid of 32 patches as direct whole-slide segmentation is computationally infeasible. The scripts below are listed in pipeline order:

1. **`reseg.py`** - run once per patch (32 total, `sys.argv[1]` selects the patch index). For a given patch: crops the region from the full Zarr, runs Cellpose segmentation (`cyto3` model) on the DAPI channel, then runs Proseg transcript-informed refinement (via `proseg_wrapper.py`) on that patch, and writes results back into a per-patch SpatialData Zarr.

2. **`seams_flow.py`** - this is based on `reseg.py`, and only computes seam flow fields - no masks. For each of 52 seam regions (28 horizontal + 24 vertical strips centred on patch boundaries), runs Cellpose to extract flow fields and cell probabilities.

3. **`seam_mask_generation.py`** - combines the 52 seam flow fields onto one global canvas (distance-weighted blending, down-weighting estimates near patch edges), then runs Cellpose's `compute_masks` once over this blended canvas to produce a single continuous ("lattice") mask spanning all patch boundaries.

4. **`merge_masks_full.py`** - merges and deduplicates the 32 per-patch Cellpose masks (from `reseg.py`) with the lattice mask (from `seam_mask_generation.py`), using IoU ≥ 0.3 to identify duplicate instances and keeping whichever instance's centroid sits furthest from a patch edge. Produces the final merged whole-slide Cellpose segmentation.

5. **`proseg_wsi.py`** - takes the merged whole-slide Cellpose mask (`merged_masks.npy`) as a prior and re-tiles the slide into 117 overlapping tiles (3000 px core + 300 px halo) for whole-slide Proseg refinement using raw Xenium transcripts as evidence. Shape and transcript-count quality filters are applied only at the final assembly step. Estimated runtime ~26 hours on a compute node.

**`proseg_wrapper.py`** - shared helper module imported by `reseg.py` and `seams_flow.py` (Proseg refinement, Zarr metadata fixes, SpatialData integration, visualisation and comparison utilities). 
