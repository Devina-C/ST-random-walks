#!/usr/bin/env python3
"""
decoy_specificity_check.py
==========================
Reads the labelled per-LR-pair summary from a decoy_random run and answers the
only question that matters: of the random (decoy) pairs, what fraction were
called significant?

  - Real (CellChat) pairs should be ~mostly significant (your ~0.97).
  - Decoy (random) pairs should be ~rarely significant (near the nominal FDR, q).

If the decoy rate is near q  -> the test discriminates; your real result is trustworthy.
If the decoy rate is high    -> the null is detecting generic spatial autocorrelation,
                                not specific signalling; fix the permutation scheme.

PREREQUISITE
------------
The permutation output must carry a `pair_type` column (the one-line add from the
modified pair-generation block: "pair_type": row["pair_type"]). This script will
read that label from the summary CSV if present, or recover it from the raw
per-cell parquet chunks if the summary doesn't carry it.

USAGE
-----
    python decoy_specificity_check.py --r 0.7
    python decoy_specificity_check.py --r 0.7 --q 0.05 \
        --summary /path/to/percell_BH_twostage_summary_r0700.csv
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

# ─── paths (match your pipeline) ────────────────────────────────────
BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
RESULTS_DIR = os.path.join(BASE_DIR, "results/ccc_results")
PERCELL_DIR = os.path.join(BASE_DIR, "results/permutation_percell")

# ─── args ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--r", type=float, default=0.7, help="RWR restart probability")
ap.add_argument("--q", type=float, default=0.05, help="nominal FDR / target decoy rate")
ap.add_argument("--summary", type=str, default=None,
                help="path to per-LR summary CSV (auto-detected if omitted)")
ap.add_argument("--sig_col", type=str, default="pair_significant",
                help="boolean column marking a significant family")
ap.add_argument("--p_col", type=str, default="simes_p",
                help="pair-level p-value column (for the uniformity check)")
ap.add_argument("--outdir", type=str, default=None,
                help="where to write the figure (default: alongside summary)")
args = ap.parse_args()

R_TAG = f"r{int(args.r * 1000):04d}"
Q = args.q

# ─── helpers ────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion (better than normal at extremes)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def recover_pair_type_from_parquet(summary):
    """If the summary lacks pair_type, pull a (sender,lig,rec)->pair_type map
    from the raw per-cell parquet, which carries the label."""
    pattern = os.path.join(PERCELL_DIR, R_TAG, "percell_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"ERROR: no pair_type in summary and no raw parquet at {pattern}.\n"
                 f"Make sure the permutation output includes the pair_type column.")
    print(f"  pair_type not in summary; recovering from {len(files)} parquet chunks...")
    keys = ["sender_ct", "ligand", "receptor", "pair_type"]
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f, columns=keys).drop_duplicates())
        except (ValueError, KeyError):
            sys.exit("ERROR: raw parquet has no pair_type column either.\n"
                     "Re-run the permutation step with the labelled pair list.")
    mapping = pd.concat(parts, ignore_index=True).drop_duplicates()
    return summary.merge(mapping, on=["sender_ct", "ligand", "receptor"], how="left")


# ─── load summary ───────────────────────────────────────────────────
summary_path = args.summary or os.path.join(
    RESULTS_DIR, f"percell_BH_twostage_summary_{R_TAG}.csv")
if not os.path.exists(summary_path):
    sys.exit(f"ERROR: summary not found: {summary_path}\n"
             f"Pass it explicitly with --summary.")

print(f"Loading summary: {summary_path}")
df = pd.read_csv(summary_path)

if args.sig_col not in df.columns:
    sys.exit(f"ERROR: significance column '{args.sig_col}' not in summary. "
             f"Columns: {list(df.columns)}")

if "pair_type" not in df.columns:
    df = recover_pair_type_from_parquet(df)

df["pair_type"] = df["pair_type"].fillna("unlabelled")
df[args.sig_col] = df[args.sig_col].astype(bool)

# normalise label names: anything not 'real' is treated as decoy/background
df["group"] = np.where(df["pair_type"].str.lower().eq("real"), "real", "decoy")

print(f"  families loaded: {len(df):,}")
print(f"  label counts: {df['group'].value_counts().to_dict()}")
if (df["group"] == "decoy").sum() == 0:
    sys.exit("ERROR: no decoy families found. Did you run with --lr_mode decoy_random?")

# ─── headline: significance rate per group (family level) ───────────
print("\n" + "=" * 64)
print(f"SPECIFICITY CHECK  (r={args.r}, target decoy rate q={Q})")
print("=" * 64)

rates = {}
for grp in ["real", "decoy"]:
    sub = df[df["group"] == grp]
    k = int(sub[args.sig_col].sum())
    n = len(sub)
    p, lo, hi = wilson_ci(k, n)
    rates[grp] = (p, lo, hi, k, n)
    print(f"  {grp:6s}: {p*100:5.1f}% significant   "
          f"({k:,}/{n:,}, 95% CI [{lo*100:.1f}, {hi*100:.1f}])")

# unique-pair view (a pair counts significant if sig in >=1 sender)
print("\n  [unique-pair view: significant in >=1 sender]")
for grp in ["real", "decoy"]:
    sub = df[df["group"] == grp]
    per_pair = sub.groupby(["ligand", "receptor"])[args.sig_col].any()
    k, n = int(per_pair.sum()), len(per_pair)
    p, lo, hi = wilson_ci(k, n)
    print(f"  {grp:6s}: {p*100:5.1f}% of unique pairs  "
          f"({k:,}/{n:,}, 95% CI [{lo*100:.1f}, {hi*100:.1f}])")

# ─── decoy p-value uniformity (supplementary diagnostic) ────────────
decoy_p = df.loc[df["group"] == "decoy", args.p_col].dropna().values
ks_line = ""
if len(decoy_p) > 0:
    try:
        from scipy.stats import kstest
        ks = kstest(decoy_p, "uniform")
        ks_line = (f"\n  decoy p-value uniformity (KS vs Uniform[0,1]): "
                   f"D={ks.statistic:.3f}, p={ks.pvalue:.3g}")
        print(ks_line.replace("\n  ", "  "))
    except ImportError:
        pass

# ─── verdict ────────────────────────────────────────────────────────
decoy_rate = rates["decoy"][0]
print("\n" + "-" * 64)
if decoy_rate <= 2 * Q:
    verdict = "PASS"
    msg = ("decoys rejected as expected. The test discriminates signal from "
           "junk; the real-pair rate is trustworthy.")
elif decoy_rate <= 0.30:
    verdict = "MARGINAL"
    msg = ("decoy rate is above nominal. Some inflation present — the null is "
           "partly picking up spatial structure rather than specific signalling.")
else:
    verdict = "FAIL"
    msg = ("decoys are being called significant at a high rate. The all-cells "
           "shuffle is detecting generic spatial autocorrelation, so the 97% on "
           "real pairs is not meaningful. Fix the permutation scheme "
           "(e.g. within-receiver-cell-type or distance-preserving swaps).")
print(f"VERDICT: {verdict}  (decoy rate {decoy_rate*100:.1f}%, target ~{Q*100:.0f}%)")
print(f"         {msg}")
print("-" * 64)

# ─── figure ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = args.outdir or os.path.dirname(os.path.abspath(summary_path))
os.makedirs(outdir, exist_ok=True)
fig_path = os.path.join(outdir, f"decoy_specificity_{R_TAG}.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# panel 1: significance rates with Wilson CIs
groups = ["real", "decoy"]
ys = [rates[g][0] * 100 for g in groups]
errs = [[ (rates[g][0] - rates[g][1]) * 100 for g in groups],
        [ (rates[g][2] - rates[g][0]) * 100 for g in groups]]
colors = ["#2a7", "#c44"]
axes[0].bar(groups, ys, yerr=errs, capsize=6, color=colors, alpha=0.85)
axes[0].axhline(Q * 100, ls="--", color="0.4", lw=1)
axes[0].text(1.4, Q * 100 + 2, f"target {Q*100:.0f}%", color="0.4", fontsize=9)
axes[0].set_ylabel("% families significant")
axes[0].set_ylim(0, 105)
axes[0].set_title("Significance rate: real vs decoy")
for i, g in enumerate(groups):
    axes[0].text(i, ys[i] + 3, f"{ys[i]:.1f}%", ha="center", fontsize=10)

# panel 2: pair-level p-value histograms
bins = np.linspace(0, 1, 26)
for g, c in zip(groups, colors):
    pv = df.loc[df["group"] == g, args.p_col].dropna().values
    if len(pv):
        axes[1].hist(pv, bins=bins, density=True, histtype="step",
                     lw=2, color=c, label=f"{g} (n={len(pv)})")
axes[1].axhline(1.0, ls=":", color="0.5", lw=1)  # uniform reference
axes[1].set_xlabel(f"pair-level p ({args.p_col})")
axes[1].set_ylabel("density")
axes[1].set_title("p-value distribution\n(decoys should be ~flat = uniform)")
axes[1].legend(fontsize=8)

# panel 3: ECDF of p-values
for g, c in zip(groups, colors):
    pv = np.sort(df.loc[df["group"] == g, args.p_col].dropna().values)
    if len(pv):
        axes[2].step(pv, np.arange(1, len(pv) + 1) / len(pv), color=c, lw=2, label=g)
axes[2].plot([0, 1], [0, 1], ls=":", color="0.5", lw=1, label="uniform")
axes[2].set_xlabel(f"pair-level p ({args.p_col})")
axes[2].set_ylabel("cumulative fraction")
axes[2].set_title("ECDF (decoys should track the diagonal)")
axes[2].legend(fontsize=8)

fig.suptitle(f"Decoy specificity check  —  {R_TAG}  —  VERDICT: {verdict}",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"\nSaved figure: {fig_path}")