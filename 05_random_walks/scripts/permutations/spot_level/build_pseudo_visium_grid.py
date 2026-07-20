#!/usr/bin/env python3
"""
build_pseudo_visium_grid.py
============================
Bins single-cell (Xenium) data onto a hexagonal grid matching real 10x
Visium geometry -- 55um spot diameter, 100um center-to-center spacing --
aggregates ligand/receptor expression per spot, and writes:

    results/pseudo_visium_adata.h5ad   spot-level AnnData (X = summed
                                        counts per spot, obs['n_cells']
                                        = number of cells pooled into
                                        each spot, obsm['spatial'] =
                                        spot centroid coordinates)
    results/pseudo_visium_graph.pkl    networkx.Graph on spot centroids,
                                        same format as
                                        results/disparity_graph.pkl /
                                        results/knn_graph.pkl, so it's a
                                        drop-in GRAPH_CACHE

IMPORTANT CAVEATS
------------------------------------------------------------------------
   fitted lambda (~5-6um) is below the 100um spot spacing -
   binning at this scale discards the sub-cellular diffusion structure
   the RWR framework is built to resolve

Usage:
    python build_pseudo_visium_grid.py --spot_diameter 55 --spot_spacing 100
"""
import argparse, json, os, pickle, time
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from shapely.geometry import Point, Polygon as ShapelyPolygon

BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH  = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH    = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
OUT_ADATA   = os.path.join(BASE_DIR, "results/pseudo_visium_adata.h5ad")
OUT_GRAPH   = os.path.join(BASE_DIR, "results/pseudo_visium_graph.pkl")

ap = argparse.ArgumentParser()
ap.add_argument("--spot_diameter", type=float, default=55.0,
                 help="Spot diameter in the same units as adata.obsm['spatial'] "
                      "(um for Xenium). A cell is assigned to a spot only if "
                      "it falls within spot_diameter/2 of that spot's centroid "
                      "-- cells in the gaps between spots (real Visium has "
                      "gaps at 55um/100um) are dropped, matching real Visium "
                      "capture geometry rather than force-assigning every cell.")
ap.add_argument("--spot_spacing", type=float, default=100.0,
                 help="Center-to-center spacing between adjacent spots. "
                      "Real Visium: 100um for a 55um-diameter spot.")
ap.add_argument("--min_cells_per_spot", type=int, default=1,
                 help="Drop spots with fewer than this many captured cells.")
ap.add_argument("--k", type=int, default=6,
                 help="k for the spot-level KNN graph (SpaFlow default is 6).")
args = ap.parse_args()

print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
adata = sc.read_h5ad(ADATA_PATH)
with open(ROI_PATH) as f:
    roi = json.load(f)
poly = ShapelyPolygon(roi["features"][0]["geometry"]["coordinates"][0])
mask = np.array([poly.contains(Point(x, y)) for x, y in adata.obsm["spatial"]])
adata = adata[mask].copy()

n = adata.n_obs
pos = adata.obsm["spatial"]
print(f"Cells in ROI: {n:,}")

# 1: lay out hexagonal grid of spot centres over the tissue 
# Standard Visium hex packing: rows spaced spot_spacing * sqrt(3)/2 apart
# vertically, alternate rows offset by spot_spacing/2 horizontally.
print(f"[{time.strftime('%H:%M:%S')}] Laying out hex grid "
      f"(spacing={args.spot_spacing}um, diameter={args.spot_diameter}um)...")

x_min, y_min = pos.min(axis=0)
x_max, y_max = pos.max(axis=0)

row_spacing = args.spot_spacing * np.sqrt(3) / 2
row_ys = np.arange(y_min, y_max + row_spacing, row_spacing)

centres = []
for r, y in enumerate(row_ys):
    x_offset = (args.spot_spacing / 2) if (r % 2 == 1) else 0.0
    row_xs = np.arange(x_min - x_offset, x_max + args.spot_spacing, args.spot_spacing) + x_offset
    for x in row_xs:
        centres.append((x, y))
centres = np.array(centres)
print(f"  Candidate spot centres before occupancy filter: {len(centres):,}")

# 2: assign each cell to the nearest centre within spot radius 
from scipy.spatial import cKDTree
tree = cKDTree(centres)
dist, spot_idx = tree.query(pos, k=1)
in_spot = dist <= (args.spot_diameter / 2)

print(f"  Cells captured by a spot: {in_spot.sum():,} / {n:,} "
      f"({100*in_spot.mean():.1f}%)")

cell_spot = np.full(n, -1, dtype=np.int64)
cell_spot[in_spot] = spot_idx[in_spot]

occupied, counts = np.unique(cell_spot[cell_spot >= 0], return_counts=True)
keep_spots = occupied[counts >= args.min_cells_per_spot]
print(f"  Occupied spots (>= {args.min_cells_per_spot} cell(s)): {len(keep_spots):,}")

spot_remap = {old: new for new, old in enumerate(sorted(keep_spots))}
n_spots = len(spot_remap)

# 3: aggregate expression per spot (sum, matching Visium's 
# multi-cell per spot capture) 
print(f"[{time.strftime('%H:%M:%S')}] Aggregating expression into "
      f"{n_spots:,} spots...")

X = adata.X
if sp.issparse(X):
    X = X.tocsr()

spot_of_cell = np.array([spot_remap.get(s, -1) for s in cell_spot])
valid = spot_of_cell >= 0

# build a (n_cells x n_spots) indicator matrix and sum via sparse matmul
rows = np.where(valid)[0]
cols = spot_of_cell[valid]
data = np.ones(len(rows))
indicator = sp.csr_matrix((data, (rows, cols)), shape=(n, n_spots))

X_spot = indicator.T @ (X if sp.issparse(X) else sp.csr_matrix(X))
X_spot = sp.csr_matrix(X_spot)

n_cells_per_spot = np.asarray(indicator.sum(axis=0)).ravel()

spot_centroids = np.zeros((n_spots, 2))
for old, new in spot_remap.items():
    m = cell_spot == old
    spot_centroids[new] = pos[m].mean(axis=0)

# majority cell type per spot - kept as obs metadata rather than used for
# aggregation itself
if "cell_type" in adata.obs.columns:
    ct = adata.obs["cell_type"].values
    dominant_ct = []
    for old, new in sorted(spot_remap.items(), key=lambda kv: kv[1]):
        m = cell_spot == old
        vals, cnts = np.unique(ct[m], return_counts=True)
        dominant_ct.append(vals[cnts.argmax()])
else:
    dominant_ct = [None] * n_spots

spot_adata = ad.AnnData(
    X=X_spot,
    obs=pd.DataFrame({
        "n_cells": n_cells_per_spot,
        "dominant_cell_type": dominant_ct,
    }),
    var=adata.var.copy(),
)
spot_adata.obsm["spatial"] = spot_centroids

os.makedirs(os.path.dirname(OUT_ADATA), exist_ok=True)
spot_adata.write_h5ad(OUT_ADATA)
print(f"  Saved: {OUT_ADATA}")
print(f"  Mean cells/spot: {n_cells_per_spot.mean():.1f} "
      f"(min {n_cells_per_spot.min():.0f}, max {n_cells_per_spot.max():.0f})")

# 4: build spot-level KNN graph, same format as knn_build.py
print(f"[{time.strftime('%H:%M:%S')}] Building spot-level KNN graph "
      f"(k={args.k})...")
D = kneighbors_graph(spot_centroids, n_neighbors=min(args.k, n_spots - 1),
                      mode="distance", include_self=False, n_jobs=-1)
D.data = np.where(D.data > 0, 1.0 / D.data, 0.0)
W = D.maximum(D.T).tocsr()
W.eliminate_zeros()

G = nx.from_scipy_sparse_array(W, edge_attribute="weight")
missing = set(range(n_spots)) - set(G.nodes())
if missing:
    G.add_nodes_from(missing)

print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
print(f"  Mean degree: {2 * G.number_of_edges() / n_spots:.1f}")

with open(OUT_GRAPH, "wb") as f:
    pickle.dump(G, f)
print(f"  Saved: {OUT_GRAPH}")

print(f"\nDone.")
