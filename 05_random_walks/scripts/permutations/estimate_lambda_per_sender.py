#!/usr/bin/env python3
# estimate lambda
# obtain per sender lambda values for r=0.7

# Estimate lambda (morphogen gradient length scale) per sender cell type
# by fitting exponential decay of RWR output field L* vs distance from senders.
# Produces per-sender lambda values for use in BH correction (n_eff calculation).
# Run once per r value; hardcode outputs into BH correction scripts.
# RWR_R must match the permutation test run.

import pickle, json, numpy as np, scanpy as sc
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu
import scipy.sparse as sp, networkx as nx

BASE   = "/scratch/users/k22026807/masters/project/random_walks"
RWR_R  = 0.7

adata = sc.read_h5ad(f"{BASE}/../celltyping/celltype_output/BC_prime/refined_annotations.h5ad")
import shapely.geometry as sg
with open(f"{BASE}/../alignment/region1_xenium.geojson") as f:
    roi = __import__('json').load(f)
poly = sg.Polygon(roi['features'][0]['geometry']['coordinates'][0])
mask = np.array([poly.contains(sg.Point(x,y)) for x,y in adata.obsm['spatial']])
adata = adata[mask].copy()

pos = adata.obsm['spatial']
tree = cKDTree(pos)
nn, _ = tree.query(pos, k=2)
dx = float(np.median(nn[:, 1]))
print(f"dx = {dx:.3f} µm")

with open(f"{BASE}/results/disparity_graph.pkl", 'rb') as f:
    G = pickle.load(f)
if isinstance(G, nx.Graph):
    missing = set(range(adata.n_obs)) - set(G.nodes())
    if missing:
        G.add_nodes_from(missing)
        print(f"  Added {len(missing)} isolated nodes")
    G = nx.to_scipy_sparse_array(G, nodelist=list(range(adata.n_obs)), format='csr')
G = G.tocsr().astype(np.float64)
row_sums = np.asarray(G.sum(axis=1)).ravel()
row_sums[row_sums == 0] = 1.0
P = sp.diags(1/row_sums) @ G
A = (sp.eye(adata.n_obs, format='csc') - (1 - RWR_R) * P.tocsc())
lu = splu(A)

cell_types = adata.obs['cell_type'].values
for sender in sorted(set(cell_types)):
    idx = np.where(cell_types == sender)[0]
    q = np.zeros(adata.n_obs); q[idx] = 1.0/len(idx)
    L_star = (RWR_R * lu.solve(q)).astype(np.float32)

    seed_pos = pos[idx]
    kd = cKDTree(seed_pos)
    d, _ = kd.query(pos, k=1)
    mask2 = (d > 0) & (L_star > L_star.max() * 1e-4)
    if mask2.sum() < 100:
        print(f"{sender:35s} | too few points"); continue
    slope, _ = np.polyfit(d[mask2], np.log(L_star[mask2]), 1)
    lam = -1.0/slope if slope < 0 else None
    print(f"{sender:35s} | n_seeds={len(idx):5d} | λ = {lam:.2f} µm")