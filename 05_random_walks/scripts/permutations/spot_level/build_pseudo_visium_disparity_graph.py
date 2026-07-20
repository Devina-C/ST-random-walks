#!/usr/bin/env python3
"""
build_pseudo_visium_disparity_graph.py
========================================
Builds a disparity-filtered backbone graph on the pseudo-Visium spot
centroids produced by build_pseudo_visium_grid.py, using the same
network-backboning method (Serrano, Boguna & Vespignani, PNAS 2009) as
results/disparity_graph.pkl at the single-cell level -- so the spot-level
comparison isn't confounded by also switching graph-construction method.

METHOD
------
1. Candidate graph: Delaunay triangulation of spot centroids, weighted
   by inverse distance. This is the standard initial graph for spatial
   disparity filtering -- it gives every spot a natural, roughly
   uniform set of candidate neighbours (the same role a fully-connected
   or KNN candidate graph plays for non-spatial disparity filtering)
   without an arbitrary k choice.
2. Disparity filter: for each node i with degree k_i and normalised
   edge weights p_ij = w_ij / sum_j(w_ij), the edge (i,j) gets a
   p-value alpha_ij = (1 - p_ij)^(k_i - 1) under the null that weights
   at node i are placed uniformly at random on its k_i edges (Serrano
   et al., Eq. 2). An edge survives into the backbone if alpha_ij is
   below the significance threshold from EITHER endpoint's test (the
   filter is evaluated independently per node, then edges are unioned).
3. Degree-1 candidate nodes are exempted from the test (k-1=0 makes the
   formula degenerate) and their single edge is always kept, matching
   the standard implementation.

Output format matches disparity_graph.pkl / knn_graph.pkl: networkx.Graph,
integer nodes 0..n_spots-1 matching pseudo_visium_adata.h5ad row order,
'weight' edge attribute = inverse distance. Drop-in GRAPH_CACHE for
fit_sbm_null.py / permutation_test_percell_sbm.py.

Usage:
    python build_pseudo_visium_disparity_graph.py --alpha 0.05
"""
import argparse, os, pickle, time
import numpy as np
import networkx as nx
import scanpy as sc
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

BASE_DIR   = "/scratch/users/k22026807/masters/project/random_walks"
SPOT_ADATA = os.path.join(BASE_DIR, "results/pseudo_visium_adata.h5ad")
OUT_PATH   = os.path.join(BASE_DIR, "results/pseudo_visium_disparity_graph.pkl")

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.05,
                 help="Disparity filter significance threshold. Lower = "
                      "sparser backbone (fewer edges survive).")
args = ap.parse_args()

print(f"[{time.strftime('%H:%M:%S')}] Loading pseudo-Visium spot data...")
spot_adata = sc.read_h5ad(SPOT_ADATA)
pos = spot_adata.obsm["spatial"]
n = spot_adata.n_obs
print(f"Spots: {n:,}")

# ── STEP 1: Delaunay candidate graph, inverse-distance weights ────────
print(f"[{time.strftime('%H:%M:%S')}] Building Delaunay candidate graph...")
tri = Delaunay(pos)
edges = set()
for simplex in tri.simplices:
    for a in range(3):
        for b in range(a + 1, 3):
            i, j = simplex[a], simplex[b]
            edges.add((min(i, j), max(i, j)))
print(f"  Delaunay candidate edges: {len(edges):,}")

G_candidate = nx.Graph()
G_candidate.add_nodes_from(range(n))
for i, j in edges:
    d = np.linalg.norm(pos[i] - pos[j])
    if d > 0:
        G_candidate.add_edge(i, j, weight=1.0 / d)

print(f"  Mean candidate degree: {2 * G_candidate.number_of_edges() / n:.1f}")


# ── STEP 2: disparity filter ───────────────────────────────────────────
def disparity_filter(G, alpha):
    """
    Serrano, Boguna & Vespignani (2009) disparity filter.
    Returns a new graph containing only edges with p-value < alpha
    from at least one endpoint's local test.
    """
    backbone = nx.Graph()
    backbone.add_nodes_from(G.nodes())
    n_tested, n_kept = 0, 0

    for i in G.nodes():
        neighbours = list(G[i])
        k_i = len(neighbours)

        if k_i == 0:
            continue
        if k_i == 1:
            # degenerate case: (k_i - 1) = 0, the test formula is undefined
            # (every edge trivially "significant" for a degree-1 node) --
            # keep the single edge, standard convention.
            j = neighbours[0]
            backbone.add_edge(i, j, weight=G[i][j]["weight"])
            continue

        s_i = sum(G[i][jj]["weight"] for jj in neighbours)
        for j in neighbours:
            n_tested += 1
            w_ij = G[i][j]["weight"]
            p_ij = w_ij / s_i
            alpha_ij = (1.0 - p_ij) ** (k_i - 1)
            if alpha_ij < alpha:
                n_kept += 1
                backbone.add_edge(i, j, weight=w_ij)

    print(f"  Directed edge-endpoint tests: {n_tested:,}  "
          f"Passed (per-endpoint): {n_kept:,}")
    return backbone


print(f"[{time.strftime('%H:%M:%S')}] Applying disparity filter "
      f"(alpha={args.alpha})...")
G = disparity_filter(G_candidate, args.alpha)

isolated = sum(1 for i in G.nodes() if G.degree(i) == 0)
print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
print(f"  Isolated nodes after filtering: {isolated:,} "
      f"({100*isolated/n:.1f}%) -- these spots will have zero diffusion "
      f"neighbours; check whether that's expected (e.g. tissue-edge "
      f"spots) or a sign alpha is too strict for this spot density.")
print(f"  Mean backbone degree: {2 * G.number_of_edges() / n:.1f}")

with open(OUT_PATH, "wb") as f:
    pickle.dump(G, f)
print(f"\nSaved: {OUT_PATH}")
print("\nPoint GRAPH_CACHE in fit_sbm_null.py / permutation_test_percell_sbm.py")
print("at this path (and ADATA_PATH at results/pseudo_visium_adata.h5ad) to")
print("run the same disparity-graph-based pipeline at spot resolution.")