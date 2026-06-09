#!/usr/bin/env python3
"""
run_rwr.py
==========
Run MultiXrank RWR for one sender cell type at a given restart probability.

The config.yml inside RUN_DIR controls the actual restart probability;
RWR_RESTART here only determines which r-tagged RUN_DIR to use.

Usage: python run_rwr.py malignant_cell
"""
import multixrank
import sys, os

# ── parameters ────────────────────────────────────────────────────────
RWR_RESTART = 0.7        # CHANGE PER R VALUE — keep in sync with update_seeds.py
R_TAG       = f"r{int(RWR_RESTART * 1000):04d}"

# ── paths ─────────────────────────────────────────────────────────────
BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"

SEED_LABEL  = sys.argv[1] if len(sys.argv) > 1 else "malignant_cell"
RUN_DIR     = os.path.join(BASE_DIR, f"rwr_runs/{SEED_LABEL}_{R_TAG}")

print(f"=== Running RWR for {SEED_LABEL} at r={RWR_RESTART} ({R_TAG}) ===")
print(f"Run dir: {RUN_DIR}")

if not os.path.exists(f"{RUN_DIR}/config.yml"):
    print(f"ERROR: config.yml not found at {RUN_DIR}/config.yml")
    print("Run update_seeds.py first.")
    sys.exit(1)

mxr = multixrank.Multixrank(
    config=f"{RUN_DIR}/config.yml",
    wdir=RUN_DIR)

print("Running random walk with restart...")
ranking_df = mxr.random_walk_rank()

mxr.write_ranking(ranking_df,
    path=f"{RUN_DIR}/cell_ranking.tsv",
    aggregation="nomean")

print(f"Done. Ranked {len(ranking_df):,} cells.")
print(ranking_df.head())