#!/usr/bin/env python3
"""
apply_neff_bh_perspot.py
==========================
Applies the manuscript's n_eff-corrected BH procedure (Section 5.4/5.5,
Eq. 33/39/41) to the existing spot-level permutation results, and
compares real-vs-decoy significance rates against raw p<0.05 -- this is
the FIRST time n_eff-BH has actually been run in this whole
investigation; every prior number (86%, 9.2%, 5.4%, 5.8%) used raw
uncorrected p<0.05, not the manuscript's proposed correction.

LAMBDA ESTIMATION
-------------------
The manuscript specifies lambda should be "directly estimable... by
fitting the spatial decay of the observed ligand field" (Section 2.4).
We don't have D_L/mu measured independently, so lambda is estimated
empirically per LR pair from the spatial autocorrelation of C_hat
itself: bin cell/spot pairs by distance, compute the empirical
correlation rho_k in each bin, and fit an exponential decay
rho_k ~ exp(-k*dx / (lambda/sqrt(2))) per Eq. 30 (the Gaussian-kernel
approximation used throughout the theory section for analytical
tractability).

n_eff DERIVATION
-------------------
Rather than assuming the closed-form 2D Bessel result (Eq. 39, which
requires the idealised Helmholtz Green's function and a spatially
uniform, infinite-domain graph -- not exactly satisfied here), this
script uses the manuscript's more general, assumption-light formula
(Eq. 33 in terms of the empirical rho_k directly):

    n_eff = n / (1 + 2 * sum_{k=1}^{k_max} rho_k)

summing empirical autocorrelation up to the lag where |rho_k| < 0.05,
exactly as specified in the text following Eq. 33.

BH CORRECTION
----------------
Applied WITHIN each LR pair (the manuscript's primary use case, Eq. 33)
-- p_(r) <= r*q / n_eff, using each pair's own empirically-estimated
n_eff rather than the nominal n. This is compared directly against
raw p<0.05 (uncorrected) for the same pair, split by pair_type
(real vs decoy).

Usage:
    python apply_neff_bh_perspot.py \
        --results "results/permutation_perspot_null/r0700/perspot_spaflownull_r0700_chunk*.parquet" \
        --lr_ref data/lr_pairs_decoy_random_seed2024.csv \
        --spot_spacing 100 --q 0.05
"""
import argparse, glob
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=str, required=True,
                 help="Glob pattern for the permutation result parquet files.")
ap.add_argument("--lr_ref", type=str, required=True,
                 help="CSV with ligand,receptor,pair_type columns.")
ap.add_argument("--spot_spacing", type=float, default=100.0,
                 help="Grid spacing (delta_x) for the lag-binning, in the "
                      "same units as x/y (um). Real Visium spacing = 100.")
ap.add_argument("--q", type=float, default=0.05,
                 help="Target FDR level.")
ap.add_argument("--n_lag_bins", type=int, default=15,
                 help="Number of distance bins for the autocorrelation "
                      "estimate. Kept modest since this is computed per "
                      "LR pair, potentially many times.")
ap.add_argument("--min_units_for_neff", type=int, default=30,
                 help="Skip n_eff estimation (fall back to n_eff=n, i.e. "
                      "no correction) for LR pairs with fewer than this "
                      "many nonzero spots -- autocorrelation estimates "
                      "are unreliable on very small samples.")
args = ap.parse_args()

files = glob.glob(args.results)
print(f"Found {len(files)} result files")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Total rows: {len(df):,}")

if "pair_type" not in df.columns:
    lr_ref = pd.read_csv(args.lr_ref)
    df = df.merge(lr_ref[["ligand", "receptor", "pair_type"]],
                   on=["ligand", "receptor"], how="left")
    n_unmatched = df["pair_type"].isna().sum()
    if n_unmatched:
        print(f"WARNING: {n_unmatched} unmatched rows")
else:
    print("pair_type already present in results (SBM output) -- skipping merge")
    n_unmatched = df["pair_type"].isna().sum()
    if n_unmatched:
        print(f"WARNING: {n_unmatched} rows have missing pair_type")

df["sig_05_raw"] = df["p_value"] < 0.05


def estimate_lambda_and_neff(sub_df, spacing, n_lag_bins, min_units):
    """
    sub_df: rows for ONE LR pair (columns: x, y, c_hat/obs_ccc, p_value).
    Returns (lambda_est, n_eff, k_max) or (None, len(sub_df), 0) if the
    sample is too small to estimate autocorrelation reliably.
    """
    n_units = len(sub_df)
    if n_units < min_units:
        return None, float(n_units), 0

    x = sub_df["x"].values
    y = sub_df["y"].values
    vals = sub_df["obs_ccc"].values.astype(np.float64)
    vals = vals - vals.mean()  # centre for correlation

    coords = np.column_stack([x, y])
    # pairwise distances -- fine for a few hundred to low-thousands of
    # spots per LR pair; would need a KD-tree / neighbour cap for much
    # larger per-pair unit counts.
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(coords))

    max_dist = np.percentile(dist_mat[dist_mat > 0], 90)
    bin_edges = np.linspace(0, max_dist, n_lag_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    rho_k = []
    var_v = np.var(vals)
    if var_v == 0:
        return None, float(n_units), 0

    for b in range(1, n_lag_bins + 1):
        lo, hi = bin_edges[b - 1], bin_edges[b]
        mask = (dist_mat >= lo) & (dist_mat < hi)
        np.fill_diagonal(mask, False)
        i_idx, j_idx = np.where(mask)
        if len(i_idx) < 10:
            rho_k.append(0.0)
            continue
        cov = np.mean(vals[i_idx] * vals[j_idx])
        rho = cov / var_v
        rho_k.append(rho)

    rho_k = np.array(rho_k)

    # k_max: first lag where |rho_k| < 0.05, per the manuscript's
    # stopping criterion following Eq. 33
    below = np.where(np.abs(rho_k) < 0.05)[0]
    k_max = below[0] + 1 if len(below) > 0 else len(rho_k)

    sum_rho = rho_k[:k_max].sum()
    n_eff = n_units / (1 + 2 * max(sum_rho, 0))  # clip negative sums
    n_eff = max(n_eff, 1.0)

    # back out lambda from the fitted decay for reporting/diagnostic
    # purposes (Eq. 30: rho_k ~ exp(-k*dx / (lambda/sqrt(2)))). Only use
    # clearly-positive lags for the log-linear fit -- the lag right at
    # k_max is, by construction, close to the |rho|<0.05 crossover and
    # can be slightly negative, which would otherwise break exp decay
    # fitting entirely for a single near-zero value.
    clearly_positive = rho_k[:k_max][rho_k[:k_max] > 0.02]
    if len(clearly_positive) >= 2:
        lags = (np.where(rho_k[:k_max] > 0.02)[0] + 1) * spacing
        log_rho = np.log(clearly_positive)
        slope, _ = np.polyfit(lags, log_rho, 1)
        lambda_est = -np.sqrt(2) / slope if slope < 0 else np.nan
    else:
        lambda_est = np.nan

    return lambda_est, n_eff, k_max


def bh_correct(p_values, n_eff, q):
    """
    BH rejection using n_eff in place of nominal n (Eq. 33):
    reject p_(r) if p_(r) <= r*q/n_eff, for the largest such r.
    Returns boolean significance array in original order.
    """
    p = np.asarray(p_values)
    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, len(p) + 1)
    thresholds = ranks * q / n_eff
    passed = sorted_p <= thresholds
    if not passed.any():
        return np.zeros(len(p), dtype=bool)
    r_star = np.where(passed)[0].max() + 1
    sig_sorted = np.zeros(len(p), dtype=bool)
    sig_sorted[:r_star] = True
    sig = np.zeros(len(p), dtype=bool)
    sig[order] = sig_sorted
    return sig


results = []
pairs = df[["ligand", "receptor", "pair_type"]].drop_duplicates()
print(f"\nEstimating lambda/n_eff and applying n_eff-BH for "
      f"{len(pairs)} LR pairs...")

for i, (_, row) in enumerate(pairs.iterrows()):
    lig, rec, pt = row["ligand"], row["receptor"], row["pair_type"]
    sub = df[(df["ligand"] == lig) & (df["receptor"] == rec)].copy()

    lambda_est, n_eff, k_max = estimate_lambda_and_neff(
        sub, args.spot_spacing, args.n_lag_bins, args.min_units_for_neff
    )

    sig_neff = bh_correct(sub["p_value"].values, n_eff, args.q)

    results.append({
        "ligand": lig, "receptor": rec, "pair_type": pt,
        "n_units": len(sub), "n_eff": n_eff, "lambda_est": lambda_est,
        "k_max": k_max,
        "frac_sig_raw": sub["sig_05_raw"].mean(),
        "frac_sig_neffBH": sig_neff.mean(),
        "n_sig_raw": int(sub["sig_05_raw"].sum()),
        "n_sig_neffBH": int(sig_neff.sum()),
    })

    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(pairs)}]")

pair_results = pd.DataFrame(results)

print("\n=== Overall comparison: raw p<0.05 vs n_eff-BH, by pair_type ===")
summary = pair_results.groupby("pair_type").agg(
    mean_frac_sig_raw=("frac_sig_raw", "mean"),
    mean_frac_sig_neffBH=("frac_sig_neffBH", "mean"),
    median_n_eff=("n_eff", "median"),
    median_lambda=("lambda_est", "median"),
    n_pairs=("ligand", "count"),
)
print(summary)

out_path = "neff_bh_perspot_results.csv"
pair_results.to_csv(out_path, index=False)
print(f"\nSaved per-pair results: {out_path}")

print("\nTop 10 REAL pairs by frac_sig_neffBH:")
print(pair_results[pair_results.pair_type == "real"]
      .sort_values("frac_sig_neffBH", ascending=False).head(10).to_string(index=False))

print("\nTop 10 DECOY pairs by frac_sig_neffBH (watch for GATA3-MAPK15):")
print(pair_results[pair_results.pair_type == "decoy"]
      .sort_values("frac_sig_neffBH", ascending=False).head(10).to_string(index=False))