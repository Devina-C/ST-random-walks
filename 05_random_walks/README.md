# Random walks with restart

Implementation of the random walk with restart (RWR) 
framework for spatial cell-cell communication (CCC) inference, 
including graph-based diffusion scoring and three statistical null models 
tested for significance.

This folder was restructured to keep only the scripts that produced the results reported in the thesis. 
Exploratory and superseded variants were removed or archived locally.


## Structure

```
scripts/
├── observed_rwr/           RWR pipeline + communication scoring
├── permutations/
│   ├── decoy_specificity_check.py     (shared: real-vs-decoy comparison)
│   ├── estimate_lambda_per_sender.py  (shared: feeds λ into BH correction)
│   ├── generate_decoy_pairs.py        (shared: generates ~3000 decoy pairs)
│   ├── cell_level/          single-cell resolution nulls
│   └── spot_level/          spot-level (SpaFlow-style) null
└── validations/             not tracked in this repo (see .gitignore)
```

Note: `graph.py` (spatial graph construction - disparity filter, k-NN, Delaunay, radius graph) is in `04_network_construction/`. 
The RWR scripts use its cached output (`results/disparity_graph.pkl`).

## Core RWR pipeline (`observed_rwr/`)

- **`run_rwr_observed_all.py`** - final RWR and ligand-specific communication scoring. Computes a ligand-specific L* field for every (sender cell type, ligand) combination.
- **`run_rwr_observed_sweep.py`** - same logic as above, run across a range of restart probabilities, r.
- **`verify_pipeline_consistency.py`** - sanity check confirming `run_rwr_observed_all.py` and `permutation_test_percell_decoys.py` independently compute the same L* for a given (sender, ligand) pair, i.e. that the observed-CCC pipeline and the permutation/decoy pipeline are built on one consistent graph + seeding logic.
- **`wrapper_restart_prob.sh`** - shell wrapper for running the above across restart probabilities on the cluster.

## Statistical null models (`permutations/`)

Three null formulations were tested, each permuting cell/spot position and recomputing RWR, holding µ, D_L, α fixed.

**n_eff**: because the RWR-diffused signal is spatially smooth (correlation length λ), adjacent cells/spots give correlated test statistics rather than independent ones. Using the raw cell/spot count as the number of tests for Benjamini-Hochberg correction is therefore over-conservative. n_eff = n · Δx² / (8π · λ²) estimates the *effective* (independent) sample size and is used in place of the raw count during BH correction.


### Shared scripts

- `decoy_specificity_check.py` - computes the headline real-vs-decoy significance comparison from a labelled BH-correction summary CSV.
- `generate_decoy_pairs.py` - generates the ~3000 random decoy ligand-receptor pairs (explicitly excluding real CellChatDB entries).
- `estimate_lambda_per_sender.py` - estimates the spatial correlation length λ per sender cell type, used in the n_eff calculation.

### `cell_level/` - single-cell resolution nulls (global + within-cell-type)

- **`permutation_test_percell.py`** - test script for both the global permutation null and the within-cell-type permutation null; which null is run depends on the input configuration, producing output in `results/permutation_percell/r0700/` (global) vs `results/permutation_percell/r0700_within_type/` (within-cell-type).
- **`permutation_test_percell_decoys.py`** - decoy-pair version of the above.
- **`BH_correction_twostage.py`** - two-stage Benjamini-Hochberg correction (Simes' method at tissue level, then per-cell BH using n_eff) applied to both cell-level nulls. This is the patched version (adds a `--percell_dir` argument and a `pair_type` fallback for non-decoy runs).

### `spot_level/` - spot-level coordinate-permutation null

- `build_pseudo_visium_grid.py` - bins single cells into pseudo-Visium spots.
- `build_pseudo_visium_disparity_graph.py` - builds the disparity-filtered spatial graph over those spots.
- **`permutation_test_perspot_null.py`** - spot-level permutation test (permutes coordinates, rebuilds graph, recomputes RWR), run at B=1000.
- **`apply_neff_bh_perspot.py`** - n_eff-based correction.
