#!/usr/bin/env python3

# Two-stage BH correction (manuscript alternative to joint correction).
#
# Stage 1 — tissue level:
#     For each (sender_ct, ligand, receptor) pair, aggregate cell-level
#     p-values into a single tissue-level p-value using Simes' method.
#     Apply BH over |P| pairs at FDR level q.
#
# Stage 2 — cell level (within passing pairs only):
#     For each pair that passes stage 1, apply BH at the cell level using
#     n_eff = n * Δx² / (8π * λ²) as the effective number of tests
#     (manuscript eq 73). This reduces the family size at stage 2 to
#     only the pairs with genuine tissue-level signal.
#
# Simes' method: for a group of m p-values sorted as p_(1) ≤ ... ≤ p_(m),
#     p_Simes = min_k { m * p_(k) / k }
# This is valid under positive dependence (PRDS), which holds here because
# spatially adjacent cells share the same diffusion field.
#
# Inputs:
#     results/permutation_percell/r{R_TAG}/*.parquet     ← per-cell test output
#     results/ccc_results/permutation_test_results_r{R_TAG}.csv  ← λ values
#
# Outputs:
#     results/ccc_results/percell_BH_twostage_{R_TAG}.parquet       ← cell level
#     results/ccc_results/percell_BH_twostage_summary_{R_TAG}.csv   ← pair level
#     results/ccc_results/percell_BH_twostage_by_receiver_ct_{R_TAG}.csv

import argparse
import glob
import os
import time
import numpy as np
import pandas as pd

# ─── paths ──────────────────────────────────────────────────────────
BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
PERCELL_DIR = os.path.join(BASE_DIR, "results/permutation_percell")
RESULTS_DIR = os.path.join(BASE_DIR, "results/ccc_results")

# ─── args ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--r",  type=float, default=0.7, help="RWR restart probability")
ap.add_argument("--q",  type=float, default=0.05, help="FDR threshold")
ap.add_argument("--dx", type=float, default=6.57,
                help="Median nearest-neighbour spacing (µm)")
ap.add_argument("--label", type=str, default="",
                help="suffix for output filenames, e.g. 'decoy' -> *_twostage_decoy_r0700.*")
ap.add_argument("--in_null", type=str, default="",
                help="null-tagged input subdir, e.g. 'within_type_expr' -> reads r0700_within_type_expr/")
ap.add_argument("--percell_dir", type=str, default=PERCELL_DIR,
                help="Directory containing r{R_TAG}/*.parquet chunks "
                     "(e.g. results/permutation_percell_sbm for the SBM run).")
args = ap.parse_args()
PERCELL_DIR = args.percell_dir

R_TAG = f"r{int(args.r * 1000):04d}"
TAG   = f"{args.label}_{R_TAG}" if args.label else R_TAG
Q     = args.q
DX    = args.dx

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Step 1: load all per-cell parquet chunks ────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading per-cell chunks ({R_TAG})...")
in_dir  = f"{R_TAG}_{args.in_null}" if args.in_null else R_TAG
pattern = os.path.join(PERCELL_DIR, in_dir, "percell_*.parquet")
files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit(f"No files matching {pattern}")
print(f"  Found {len(files)} chunks")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"  Total cell-level tests:  {len(df):,}")
print(f"  Unique LR pairs:         {df.groupby(['sender_ct','ligand','receptor']).ngroups}")
print(f"  Unique senders:          {df['sender_ct'].nunique()}")

# fallback: non-decoy runs (e.g. plain SBM permutation output) have no
# pair_type column -- treat everything as 'real' so downstream grouping works
if "pair_type" not in df.columns:
    print("  No pair_type column found (non-decoy run) -- labelling all pairs 'real'")
    df["pair_type"] = "real"

# ─── Step 2: attach λ ────────────────────────────────────────────────
LAMBDA_PER_SENDER = {
    "B cell":                        5.56,
    "Endothelial cell":              5.55,
    "Epithelial cell":               4.47,
    "Fibroblast":                    5.54,
    "Malignant cell":                5.38,
    "Mast cell":                     5.45,
    "Myeloid cell":                  5.69,
    "Pericyte":                      5.57,
    "Plasmacytoid dendritic cell":   5.28,
    "T cell":                        5.87,
}

df['lambda_est'] = df['sender_ct'].map(LAMBDA_PER_SENDER)
n_missing = df['lambda_est'].isna().sum()
if n_missing > 0:
    print(f"  WARNING: {n_missing} rows with no lambda, using fallback 5.54")
    df['lambda_est'] = df['lambda_est'].fillna(5.54)
print(f"  Lambda range: {df['lambda_est'].min():.2f} – {df['lambda_est'].max():.2f} µm")

# ─── Step 3: Stage 1 — Simes tissue-level p-value per pair ──────────
print(f"\n[{time.strftime('%H:%M:%S')}] Stage 1: Simes aggregation per LR pair...")

group_key = ['sender_ct', 'ligand', 'receptor']

def simes_p(p_values):
    """Simes' combined p-value for a group of tests."""
    p = np.sort(p_values)
    m = len(p)
    return float(np.min(m * p / np.arange(1, m + 1)))

tissue = (df.groupby(group_key)['p_value']
            .apply(simes_p)
            .reset_index()
            .rename(columns={'p_value': 'simes_p'}))

ptype  = df.groupby(group_key)['pair_type'].first().reset_index()
tissue = tissue.merge(ptype, on=group_key, how='left')

n_pairs = len(tissue)
print(f"  LR pairs to test at tissue level: {n_pairs}")

tissue_sorted = tissue.sort_values('simes_p', kind='mergesort').reset_index(drop=True)
ranks_t = np.arange(1, n_pairs + 1)
thresholds_t = ranks_t * Q / n_pairs
passing_t = tissue_sorted['simes_p'].values <= thresholds_t

if passing_t.any():
    k_star_t = int(np.where(passing_t)[0].max()) + 1
else:
    k_star_t = 0

tissue_sorted['pair_significant'] = np.arange(n_pairs) < k_star_t

q_raw_t = n_pairs * tissue_sorted['simes_p'].values / ranks_t
q_adj_t = np.minimum.accumulate(q_raw_t[::-1])[::-1]
tissue_sorted['pair_q_BH'] = np.clip(q_adj_t, 0, 1).astype(np.float32)

n_pairs_sig = int(tissue_sorted['pair_significant'].sum())
print(f"  Pairs significant at tissue level: {n_pairs_sig} / {n_pairs} "
      f"({100 * n_pairs_sig / n_pairs:.1f}%)")

# ─── Step 4: Stage 2 — cell-level BH within passing pairs ───────────
print(f"\n[{time.strftime('%H:%M:%S')}] Stage 2: cell-level BH within passing pairs...")

passing_set = set(
    tissue_sorted[tissue_sorted['pair_significant']]
    .apply(lambda r: (r['sender_ct'], r['ligand'], r['receptor']), axis=1)
)

df = df.merge(
    tissue_sorted[group_key + ['simes_p', 'pair_significant', 'pair_q_BH']],
    on=group_key, how='left'
)

df['significant']  = False
df['q_value_BH']   = np.float32(np.nan)
df['n_eff']        = np.float32(np.nan)
df['n_tested']     = np.int32(0)

out_rows = []

for keys, grp in df[df['pair_significant']].groupby(group_key):
    grp = grp.copy().reset_index(drop=True)
    n_grp = len(grp)

    lam = float(grp['lambda_est'].iloc[0])
    if not np.isfinite(lam) or lam <= 0:
        lam = 5.5

    n_eff = n_grp * DX**2 / (8 * np.pi * lam**2)

    grp_sorted = grp.sort_values('p_value', kind='mergesort').reset_index(drop=True)
    p_sorted = grp_sorted['p_value'].values
    ranks_c = np.arange(1, n_grp + 1)

    thresholds_c = ranks_c * Q / n_eff
    passing_c = p_sorted <= thresholds_c

    if passing_c.any():
        k_star_c = int(np.where(passing_c)[0].max()) + 1
    else:
        k_star_c = 0

    q_raw_c = n_eff * p_sorted / ranks_c
    q_adj_c = np.minimum.accumulate(q_raw_c[::-1])[::-1]

    grp_sorted['significant'] = np.arange(n_grp) < k_star_c
    grp_sorted['q_value_BH']  = np.clip(q_adj_c, 0, 1).astype(np.float32)
    grp_sorted['n_eff']       = np.float32(n_eff)
    grp_sorted['n_tested']    = np.int32(n_grp)
    out_rows.append(grp_sorted)

failed = df[~df['pair_significant']].copy()
failed['significant'] = False
failed['q_value_BH']  = np.float32(1.0)
failed['n_eff']       = np.float32(np.nan)
failed['n_tested']    = failed.groupby(group_key)['cell_id'].transform('count').astype(np.int32)

if out_rows:
    passed_df = pd.concat(out_rows, ignore_index=True)
    result = pd.concat([passed_df, failed], ignore_index=True)
else:
    print("  No pairs passed stage 1; all cells marked non-significant.")
    result = failed

n_sig = int(result['significant'].sum())
print(f"  Cell-level significant: {n_sig:,} / {len(result):,} "
      f"({100 * n_sig / len(result):.2f}%)")

out_main = os.path.join(RESULTS_DIR, f"percell_BH_twostage_{TAG}.parquet")
result.to_parquet(out_main, compression='snappy', index=False)
print(f"\nSaved: {out_main}  ({os.path.getsize(out_main)/1e6:.1f} MB)")

summary = (result.groupby(group_key)
                 .agg(n_tested=('cell_id', 'count'),
                      n_sig=('significant', 'sum'),
                      median_SES=('SES', 'median'),
                      median_p=('p_value', 'median'),
                      simes_p=('simes_p', 'first'),
                      pair_type=('pair_type', 'first'),
                      pair_significant=('pair_significant', 'first'),
                      pair_q_BH=('pair_q_BH', 'first'),
                      lambda_est=('lambda_est', 'first'),
                      n_eff=('n_eff', 'first'))
                 .reset_index())
summary['pct_sig'] = 100 * summary['n_sig'] / summary['n_tested']

out_summary = os.path.join(RESULTS_DIR, f"percell_BH_twostage_summary_{TAG}.csv")
summary.to_csv(out_summary, index=False)
print(f"Saved: {out_summary}")

if args.label and "decoy" in args.label:
    print("Skipping by_receiver_ct CSV for decoy run (huge, not meaningful).")
else:
    ct_agg = (result.groupby([*group_key, 'cell_type'])
                    .agg(n_cells=('cell_id', 'count'),
                         n_sig_cells=('significant', 'sum'),
                         median_SES=('SES', 'median'))
                    .reset_index()
                    .rename(columns={'cell_type': 'receiver_ct'}))
    ct_agg['pct_sig'] = 100 * ct_agg['n_sig_cells'] / ct_agg['n_cells']

    out_ct = os.path.join(RESULTS_DIR, f"percell_BH_twostage_by_receiver_ct_{TAG}.csv")
    ct_agg.to_csv(out_ct, index=False)
    print(f"Saved: {out_ct}")

print(f"\n{'='*60}")
print(f"HEADLINE SUMMARY — two-stage BH (r={args.r}, q={Q})")
print(f"{'='*60}")
print(f"Stage 1 — LR pairs tested:          {n_pairs}")
print(f"Stage 1 — pairs significant:         {n_pairs_sig} "
      f"({100 * n_pairs_sig / n_pairs:.1f}%)")
print(f"Stage 2 — cell-level significant:    {n_sig:,} / {len(result):,} "
      f"({100 * n_sig / len(result):.2f}%)")
print(f"Median n_eff (passing pairs):        "
      f"{summary.loc[summary['pair_significant'], 'n_eff'].median():.0f}")
print(f"Median λ:                            "
      f"{summary['lambda_est'].median():.2f} µm")

top = (summary[summary['pair_significant']]
       .nlargest(15, 'n_sig')
       [['sender_ct', 'ligand', 'receptor', 'n_sig', 'pct_sig',
         'median_SES', 'simes_p', 'pair_q_BH', 'lambda_est', 'n_eff']])
print(f"\nTop 15 LR pairs (passing stage 1) by # significant cells:")
print(top.to_string(index=False))
