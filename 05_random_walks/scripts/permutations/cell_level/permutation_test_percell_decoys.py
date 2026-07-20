#!/usr/bin/env python3

# spatial permutation test for CCC scores
# graph topology is fixed

import argparse, os, pickle, time
import itertools
import numpy as np
import pandas as pd
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon
import scanpy as sc
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.linalg import splu

BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH  = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
LR_PATH     = os.path.join(BASE_DIR, "data/cellchat_full.csv")
GRAPH_CACHE = os.path.join(BASE_DIR, "results/disparity_graph.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results/permutation_percell")
ROI_PATH = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"

ap = argparse.ArgumentParser()
ap.add_argument("--sender_idx", type=int, required=True)
ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--total", type=int, default=16)
ap.add_argument("--r", type=float, default=0.7)
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
# ── decoy / pair-mode options ──────────────────────────────────────
ap.add_argument("--lr_mode",
                choices=["cellchat", "decoy_scramble", "decoy_random"],
                default="cellchat",
                help="cellchat = real pairs only; "
                     "decoy_scramble = real + re-paired decoys; "
                     "decoy_random = real + random background pairs")
ap.add_argument("--n_background", type=int, default=3000,
                help="number of random background pairs for decoy_random")
ap.add_argument("--decoy_seed", type=int, default=2024)
args = ap.parse_args()

R_TAG = f"r{int(args.r*1000):04d}"
OUT_DIR = os.path.join(RESULTS_DIR, R_TAG)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"  Total cells (pre-ROI): {adata.n_obs:,}")

with open(ROI_PATH) as f:
    roi = json.load(f)
poly = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
mask = np.array([poly.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[mask].copy()

import re
PROBE_RX = re.compile(
    r'^(Human|Mouse|BLANK|NegControl|NegControlProbe|NegControlCodeword'
    r'|antisense|UnassignedCodeword|DeprecatedCodeword|Intergenic)',
    re.IGNORECASE)
keep_genes = [g for g in adata.var_names if not PROBE_RX.match(g)]
n_drop = adata.n_vars - len(keep_genes)
adata = adata[:, keep_genes].copy()
print(f"  Dropped {n_drop} control/mutation probes; {adata.n_vars} genes remain")

n = adata.n_obs
print(f"Cells in ROI: {n:,}")

cell_ids = adata.obs.index.values
cell_types = adata.obs["cell_type"].values
positions = adata.obsm["spatial"]

unique_senders = sorted(adata.obs["cell_type"].unique())
sender_ct = unique_senders[args.sender_idx]
print(f"Sender cell type: {sender_ct}")

# ── build the list of LR pairs to test ─────────────────────────────
gene_set = set(adata.var_names)

def genes_present(complex_name):
    return all(g in gene_set for g in complex_name.split("_"))

# real CellChat pairs, gene-filtered (original logic)
lr_df = pd.read_csv(LR_PATH, header=None, skiprows=1,
                    names=["ligand", "receptor", "pathway", "category"])
real = lr_df[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)
real = real[real["ligand"].apply(genes_present)
            & real["receptor"].apply(genes_present)].reset_index(drop=True)
real_set = set(map(tuple, real.values))
print(f"  Real CellChat pairs (gene-filtered): {len(real)}")

rng_d = np.random.default_rng(args.decoy_seed)
real2 = real.copy()
real2["pair_type"] = "real"

if args.lr_mode == "cellchat":
    lr_pairs = real2

elif args.lr_mode == "decoy_scramble":
    # same ligands & same receptors, re-paired at random -> biology broken,
    # marginal gene/expression properties preserved. Matched negative control.
    rec_shuffled = rng_d.permutation(real["receptor"].values)
    decoy = pd.DataFrame({"ligand": real["ligand"].values,
                          "receptor": rec_shuffled})
    decoy = decoy[~decoy.apply(lambda r: (r.ligand, r.receptor) in real_set, axis=1)]
    decoy = decoy[decoy["ligand"] != decoy["receptor"]].drop_duplicates()
    decoy["pair_type"] = "decoy"
    lr_pairs = pd.concat([real2, decoy], ignore_index=True)

elif args.lr_mode == "decoy_random":
    # random gene-gene pairs from the panel, excluding any real pair.
    # stand-in for "test all of them" via a representative sample.
    genes = sorted(gene_set)
    decoys = set()
    while len(decoys) < args.n_background:
        l, r = rng_d.choice(genes), rng_d.choice(genes)
        if l != r and (l, r) not in real_set:
            decoys.add((l, r))
    decoy = pd.DataFrame(list(decoys), columns=["ligand", "receptor"])
    decoy["pair_type"] = "decoy"
    lr_pairs = pd.concat([real2, decoy], ignore_index=True)

lr_pairs = lr_pairs.reset_index(drop=True)

_ok = lr_pairs["ligand"].apply(genes_present) & lr_pairs["receptor"].apply(genes_present)
if (~_ok).any():
    print(f"  Dropping {(~_ok).sum()} pair(s) referencing non-panel genes, e.g.:")
    print(lr_pairs.loc[~_ok, ["ligand", "receptor"]].head().to_string(index=False))
lr_pairs = lr_pairs[_ok].reset_index(drop=True)

print(f"  [{args.lr_mode}] total pairs to test: {len(lr_pairs)} "
      f"(real={(lr_pairs['pair_type'] == 'real').sum()}, "
      f"decoy={(lr_pairs['pair_type'] != 'real').sum()})")

# ── shard the pair list across chunks ──────────────────────────────
chunk_size = int(np.ceil(len(lr_pairs) / args.total))
start = args.chunk * chunk_size
end = min(start + chunk_size, len(lr_pairs))
lr_chunk = lr_pairs.iloc[start:end].reset_index(drop=True)

print(f"Chunk {args.chunk}/{args.total}: LR pairs [{start}:{end}]")

print(f"[{time.strftime('%H:%M:%S')}] Loading graph...")
with open(GRAPH_CACHE, "rb") as f:
    G = pickle.load(f)

if isinstance(G, nx.Graph):
    print("Graph is NetworkX; converting to scipy sparse.")
    missing = set(range(n)) - set(G.nodes())
    if missing:
        G.add_nodes_from(missing)
        print(f"Added {len(missing)} isolated nodes back")
    G = nx.to_scipy_sparse_array(
        G,
        nodelist=list(range(n)),
        format="csr",
        dtype=np.float64
    )
elif sp.issparse(G):
    G = G.tocsr()
else:
    raise TypeError(f"Unsupported graph type: {type(G)}")

assert G.shape == (n, n), f"Graph shape {G.shape} != ({n},{n})"

row_sums = np.asarray(G.sum(axis=1)).ravel()
row_sums[row_sums == 0] = 1.0
D_inv = sp.diags(1.0 / row_sums)
P = D_inv @ G

I_mat = sp.identity(n, format="csr")

# factorisation - preparing matrix for fast solving
A = (I_mat - (1.0 - args.r) * P).tocsc()

print(f"[{time.strftime('%H:%M:%S')}] Factorising RWR matrix...")
t0 = time.time()
A_lu = splu(A)
print(f"Factorisation done in {time.time() - t0:.1f}s")

def expr(gene_or_complex):
    """
    Expression vector for ligand/receptor
    Complexes are handled by multiplying subunits
    """
    genes = gene_or_complex.split("_")
    arrs = []

    for g in genes:
        x = adata[:, g].X
        x = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()
        arrs.append(x.astype(np.float32))

    if len(arrs) == 1:
        return arrs[0]

    out = arrs[0].copy()
    for arr in arrs[1:]:
        out *= arr

    return out

def build_seed(L_expr, sender_indices):
    # ligand-weighted seeds
    # normalised to sum to 1
    seed = np.zeros(n, dtype=np.float64)
    seed[sender_indices] = L_expr[sender_indices]
    s = seed.sum()
    if s > 0:
        seed /= s
    return seed

# RUN RWR
def solve_rwr(seed_vec):
    #
    return (args.r * A_lu.solve(seed_vec)).astype(np.float32)

sender_mask = cell_types == sender_ct
sender_indices = np.where(sender_mask)[0]
n_sender = len(sender_indices)

if n_sender < 10:
    print(f"Too few sender cells: {n_sender}")
    raise SystemExit(0)

print(f"N sender cells: {n_sender:,}")

# uniform seeds
#seed_obs = np.zeros(n, dtype=np.float64)
#seed_obs[sender_indices] = 1.0 / n_sender

rng = np.random.default_rng(args.seed + args.chunk * 1000 + args.sender_idx)
perms = [rng.permutation(n) for _ in range(args.B)]
perm_invs = [np.argsort(p) for p in perms]

rows = []
t_loop = time.time()

for lr_idx, row in lr_chunk.iterrows():
    lig = row["ligand"]
    rec = row["receptor"]
    pair_type = row["pair_type"]

    L_expr = expr(lig)
    R_expr = expr(rec)

    if R_expr.max() == 0 or L_expr[sender_indices].sum() == 0:
        continue

    # per-LR seed
    seed_obs = build_seed(L_expr, sender_indices)
    L_star_obs = solve_rwr(seed_obs)
    c_obs = L_star_obs * R_expr

    #L_star_obs = solve_rwr(seed_obs)
    #c_obs = L_star_obs * R_expr

    exceed = np.zeros(n, dtype=np.int32)
    null_sum = np.zeros(n, dtype=np.float64)
    null_sumsq = np.zeros(n, dtype=np.float64)

    for b in range(args.B):
        perm = perms[b]
        perm_inv = perm_invs[b]

        new_sender_indices = perm[sender_indices]

        # ligand expression follows permuted identities
        L_perm = L_expr[perm_inv]

        seed_b = build_seed(L_perm, new_sender_indices)
        L_star_b = solve_rwr(seed_b)

        R_perm_b = R_expr[perm_inv]
        c_perm_b = L_star_b * R_perm_b

        exceed += (c_perm_b >= c_obs).astype(np.int32)
        null_sum += c_perm_b
        null_sumsq += c_perm_b * c_perm_b

    null_mean = null_sum / args.B
    null_var = null_sumsq / args.B - null_mean ** 2
    null_std = np.sqrt(np.maximum(null_var, 0))
    p_value = (exceed + 1) / (args.B + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        SES = np.where(null_mean > 0, c_obs / null_mean, np.nan)

    keep = c_obs > 0

    if keep.sum() == 0:
        continue

    idx = np.where(keep)[0]

    rows.append(pd.DataFrame({
        "sender_ct": sender_ct,
        "ligand": lig,
        "receptor": rec,
        "pair_type": pair_type,
        "cell_id": cell_ids[idx],
        "cell_type": cell_types[idx],
        "x": positions[idx, 0],
        "y": positions[idx, 1],
        "obs_ccc": c_obs[idx].astype(np.float32),
        "null_mean": null_mean[idx].astype(np.float32),
        "null_std": null_std[idx].astype(np.float32),
        "p_value": p_value[idx].astype(np.float32),
        "SES": SES[idx].astype(np.float32),
    }))

    if (lr_idx + 1) % 10 == 0:
        elapsed = time.time() - t_loop
        rate = (lr_idx + 1) / elapsed
        eta = (len(lr_chunk) - lr_idx - 1) / rate
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"LR {lr_idx+1}/{len(lr_chunk)} | "
            f"{rate:.3f} pairs/s | ETA {eta/60:.1f} min"
        )

if not rows:
    print("No rows to write.")
    raise SystemExit(0)

df = pd.concat(rows, ignore_index=True)

out_name = (
    f"percell_{R_TAG}_sender{args.sender_idx:02d}_"
    f"chunk{args.chunk:02d}.parquet"
)
out_path = os.path.join(OUT_DIR, out_name)

df.to_parquet(out_path, compression="snappy", index=False)

print(f"Saved: {out_path}")
print(f"Rows: {len(df):,}")
print(f"File size: {os.path.getsize(out_path)/1e6:.1f} MB")
print(f"Total time: {(time.time() - t_loop)/60:.1f} min")