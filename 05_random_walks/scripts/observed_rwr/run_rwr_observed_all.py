#!/usr/bin/env python3
"""
run_rwr_observed_all.py
========================
Computes a genuinely LIGAND-SPECIFIC observed L* field (per the
manuscript's eq. \\ref{eq:rwr_ij}: one independent RWR solve per unique
ligand, seeded by that ligand's own expression at sender cells) for
EVERY (sender_ct, ligand) combination in one pass -- not one MultiXrank
call per pair.

Why this is fast despite being "all cells, all ligands, all senders":
    The graph itself never changes across senders or ligands -- only
    the seed vector q does. So the expensive part (factorising
    A = I - (1-r)P) is done ONCE, globally, and reused for every
    (sender, ligand) pair as a cheap triangular solve
    (`lu.solve(seed)`), exactly mirroring what
    permutation_test_percell_decoys.py already does per-pair inside
    its permutation loop -- this script is the observed-only (no
    permutation) version of that same machinery, batched over all
    senders and ligands.

    Concretely: ~1 factorisation (the expensive O(n^1.5)-ish sparse LU)
    + one solve per ligand (~O(nnz) each) instead of ~(n_senders x
    n_ligands) full re-solves or re-loads.

Replaces:
    - lr_score.py's L* loading (which reused ONE cell-type-seeded L*
      across every ligand from that sender -- not ligand-specific).
    - The MultiXrank-per-pair approach (update_seeds_ligand.py /
      run_rwr_ligand.py), which works but is much slower and has an
      unverified node-ordering assumption for the `pr=` argument.

Output: one CSV with columns
    sender_ct, ligand, receptor, receiver_ct, mean_ccc, max_ccc,
    n_receiver, pct_R_expr
-- the same schema select_lr_pair.py expects, so you can drop this
straight into that pipeline.

Usage:
    python run_rwr_observed_all.py --r 0.7
"""
import argparse, json, os, pickle, time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.linalg import splu
from shapely.geometry import Point, Polygon as ShapelyPolygon

BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH  = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH    = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
LR_PATH     = os.path.join(BASE_DIR, "data/cellchat_full.csv")
GRAPH_CACHE = os.path.join(BASE_DIR, "results/disparity_graph.pkl")
OUT_DIR     = os.path.join(BASE_DIR, "results/ccc_results")
os.makedirs(OUT_DIR, exist_ok=True)

SEED_TYPES = {
    "malignant_cell":              "Malignant cell",
    "t_cell":                      "T cell",
    "myeloid_cell":                "Myeloid cell",
    "fibroblast":                  "Fibroblast",
    "endothelial_cell":            "Endothelial cell",
    "b_cell":                      "B cell",
    "pericyte":                    "Pericyte",
    "epithelial_cell":             "Epithelial cell",
    "plasmacytoid_dendritic_cell": "Plasmacytoid dendritic cell",
    "mast_cell":                   "Mast cell",
}

SHORTLIST = {
    ("endothelial_cell", "VEGFB", "FLT1"),
    ("mast_cell",         "VEGFB", "FLT1"),
    ("fibroblast",        "VEGFB", "FLT1"),
    ("pericyte",          "PDGFA", "PDGFRB"),
    ("mast_cell",         "LGALS9","CD44"),
    ("fibroblast",        "PGF",   "FLT1"),
    ("pericyte",          "PDGFD", "PDGFRB"),
}
PERCELL_OUT_DIR = os.path.join(OUT_DIR, "percell_scores")
os.makedirs(PERCELL_OUT_DIR, exist_ok=True)


def get_expressed_ligands(adata, cell_type, ligands, min_expr=0.01):
    """Same convention as lr_score.py: ligand must be expressed in
    >=min_expr fraction of the sender cell type's cells."""
    ct_mask = (adata.obs['cell_type'] == cell_type).values
    X_ct = adata.X[ct_mask]
    if hasattr(X_ct, 'toarray'):
        X_ct = X_ct.toarray()
    else:
        X_ct = np.asarray(X_ct)
    expressed = []
    var_index = {g: i for i, g in enumerate(adata.var_names)}
    for gene in ligands:
        if gene not in var_index:
            continue
        idx = var_index[gene]
        frac = (X_ct[:, idx] > 0).mean()
        if frac >= min_expr:
            expressed.append(gene)
    return set(expressed)


def expr(adata, gene_or_complex):
    """Expression vector for ligand/receptor; complexes = product of subunits."""
    genes = gene_or_complex.split("_")
    if any(g not in adata.var_names for g in genes):
        return None
    arrs = []
    for g in genes:
        x = adata[:, g].X
        x = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()
        arrs.append(x.astype(np.float32))
    out = arrs[0].copy()
    for a in arrs[1:]:
        out *= a
    return out


def build_seed(L_expr, sender_indices, n):
    """q_i = s_i/mu restricted to sender cells, normalised to sum to 1
    (mu is an overall scale constant that cancels in the RWR solve up
    to normalisation -- matches build_seed() in
    permutation_test_percell_decoys.py exactly)."""
    seed = np.zeros(n, dtype=np.float64)
    seed[sender_indices] = L_expr[sender_indices]
    s = seed.sum()
    if s > 0:
        seed /= s
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=0.7)
    ap.add_argument("--min_expr", type=float, default=0.01,
                     help="Min fraction of sender cells expressing a "
                          "ligand for it to be tested (default 0.01, "
                          "matches lr_score.py).")
    ap.add_argument("--k_binding", type=float, default=1.0,
                     help="Binding ratio kappa (default 1.0, matches "
                          "lr_score.py's k=1.0).")
    args = ap.parse_args()

    R_TAG = f"r{int(args.r*1000):04d}"
    out_csv = os.path.join(OUT_DIR, f"ccc_all_lr_pairs_ligandseeded_{R_TAG}.csv")

    print(f"=== run_rwr_observed_all | r={args.r} | tag={R_TAG} ===")
    print(f"Output: {out_csv}")

    print("\nLoading AnnData...")
    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    polygon = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([polygon.contains(Point(x, y))
                     for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()
    n = adata.n_obs
    cell_types = adata.obs['cell_type'].values
    print(f"ROI cells: {n:,}")

    df_lr = pd.read_csv(LR_PATH, header=0,
                        names=['ligand', 'receptor', 'pathway', 'type'])
    panel_genes = set(adata.var_names)
    df_lr = df_lr[df_lr['ligand'].isin(panel_genes)].copy()
    print(f"CellChatDB pairs with ligand in panel: {len(df_lr):,}")

    # ── the ONE expensive step: load graph, factorise ONCE ─────────────
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading cached graph...")
    with open(GRAPH_CACHE, "rb") as f:
        G = pickle.load(f)

    if isinstance(G, nx.Graph):
        missing = set(range(n)) - set(G.nodes())
        if missing:
            G.add_nodes_from(missing)
        G = nx.to_scipy_sparse_array(G, nodelist=list(range(n)),
                                      format="csr", dtype=np.float64)
    elif sp.issparse(G):
        G = G.tocsr()
    else:
        raise TypeError(f"Unsupported graph type: {type(G)}")
    assert G.shape == (n, n), f"Graph shape {G.shape} != ({n},{n})"

    row_sums = np.asarray(G.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ G
    A = (sp.identity(n, format="csr") - (1.0 - args.r) * P).tocsc()

    print(f"[{time.strftime('%H:%M:%S')}] Factorising RWR matrix "
          f"(ONE factorisation, reused for every sender x ligand pair)...")
    t0 = time.time()
    A_lu = splu(A)
    print(f"  Factorisation done in {time.time()-t0:.1f}s")

    def solve_rwr(seed_vec):
        return (args.r * A_lu.solve(seed_vec)).astype(np.float32)

    # ── loop: for each sender, for each expressed ligand, ONE cheap solve ──
    results = []
    t_loop = time.time()
    n_solves = 0

    for seed_label, seed_ct in SEED_TYPES.items():
        sender_mask = (cell_types == seed_ct)
        sender_indices = np.where(sender_mask)[0]
        n_sender = len(sender_indices)
        if n_sender == 0:
            print(f"\nSender: {seed_ct} -- 0 cells, skipping")
            continue

        all_ligands = set(df_lr['ligand'].tolist())
        expressed_ligands = get_expressed_ligands(
            adata, seed_ct, all_ligands, args.min_expr)
        lr_this = df_lr[df_lr['ligand'].isin(expressed_ligands)].copy()

        print(f"\nSender: {seed_ct} (n={n_sender:,}) -- "
              f"{len(expressed_ligands):,} expressed ligands, "
              f"{len(lr_this):,} LR pairs to test")

        # Cache L_star per ligand so if the same ligand appears with
        # multiple receptors we don't re-solve the RWR for it.
        L_star_cache = {}

        for _, row in lr_this.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            pathway = row['pathway']

            if ligand not in L_star_cache:
                L_expr = expr(adata, ligand)
                if L_expr is None or L_expr[sender_indices].sum() == 0:
                    L_star_cache[ligand] = None
                else:
                    seed = build_seed(L_expr, sender_indices, n)
                    L_star_cache[ligand] = solve_rwr(seed)
                    n_solves += 1

            L_star = L_star_cache[ligand]
            if L_star is None:
                continue

            R = expr(adata, receptor)
            if R is None:
                continue
            c_hat = args.k_binding * L_star * R
            if c_hat.sum() == 0:
                continue

            if (seed_label, ligand, receptor) in SHORTLIST:
                percell_df = pd.DataFrame({
                    'cell_id': adata.obs_names,
                    'cell_type': cell_types,
                    'x': adata.obsm['spatial'][:, 0],
                    'y': adata.obsm['spatial'][:, 1],
                    'L_star': L_star,
                    'R_expr': R,
                    'c_hat': c_hat,
                })
                percell_out = os.path.join(
                    PERCELL_OUT_DIR,
                    f"percell_{ligand}_{receptor}_{seed_label}_{R_TAG}.csv")
                percell_df.to_csv(percell_out, index=False)
                print(f"    [saved per-cell scores: {percell_out}]")

            for recv_ct in np.unique(cell_types):
                recv_mask = (cell_types == recv_ct)
                if recv_mask.sum() == 0:
                    continue
                results.append({
                    'sender_ct':   seed_ct,
                    'seed_label':  seed_label,
                    'ligand':      ligand,
                    'receptor':    receptor,
                    'pathway':     pathway,
                    'receiver_ct': recv_ct,
                    'mean_ccc':    float(c_hat[recv_mask].mean()),
                    'max_ccc':     float(c_hat[recv_mask].max()),
                    'n_receiver':  int(recv_mask.sum()),
                    'pct_R_expr':  float((R[recv_mask] > 0).mean() * 100),
                })

        elapsed = time.time() - t_loop
        print(f"  [{elapsed/60:.1f} min elapsed, {n_solves} RWR solves so far]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)
    print(f"\nTotal RWR solves: {n_solves:,} "
          f"(one per unique (sender, ligand) pair with nonzero seed -- "
          f"NOT one per row in the output table)")
    print(f"Total interactions: {len(results_df):,}")
    print(f"Saved: {out_csv}")
    print(f"Total time: {(time.time()-t_loop)/60:.1f} min")

    top = results_df.sort_values('mean_ccc', ascending=False).head(30)
    print("\nTop 30 CCC interactions (genuinely ligand-specific L*):")
    print(top[['sender_ct', 'ligand', 'receptor', 'receiver_ct',
               'mean_ccc', 'pathway']].to_string())


if __name__ == '__main__':
    main()