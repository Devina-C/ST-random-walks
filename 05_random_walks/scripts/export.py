#!/usr/bin/env python3
"""
Build the disparity-filter spatial graph and export it for MultiXrank RWR.

Outputs (written to ./multixrank_input/):
    config.yml            - MultiXrank configuration
    seeds.txt             - seed cell IDs (one per line)
    network/cells.tsv     - weighted edge list (source<TAB>target<TAB>weight)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from shapely.geometry import Point, Polygon as ShapelyPolygon
from graph import disparity_filter, disparity_filter_alpha_cut

# paths 
BASE_DIR = "C:/Users/Devin/Documents/ST_ccc/05_random_walks"
ADATA_PATH = "C:/Users/Devin/Documents/ST_ccc/02_cell_typing/results/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH   = "C:/Users/Devin/Documents/ST_ccc/03_roi_extraction/results/region1_xenium.geojson"

# parameters
ALPHA          = 0.005          # disparity-filter significance threshold
RADIUS         = 200            # initial neighbour search radius (same units as spatial coords)
RWR_RESTART    = 0.7            # global restart probability for RWR


# choose seed type from command line
SEED_CELL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "Malignant cell"
SEED_LABEL = SEED_CELL_TYPE.lower().replace(' ', '_').replace('/', '_')
OUT_DIR = os.path.join(BASE_DIR, f"rwr_runs/{SEED_LABEL}")

print(f"=== Seed: {SEED_CELL_TYPE} ===")
print(f"Output dir: {OUT_DIR}")

# load AnnData and apply ROI mask 
print("Loading AnnData...")
adata = sc.read(ADATA_PATH)

with open(ROI_PATH) as f:
    roi = json.load(f)
polygon = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[mask].copy()
print(f"ROI cells: {adata.shape[0]:,}")

available = adata.obs['cell_type'].unique().tolist()
if SEED_CELL_TYPE not in available:
    print(f"ERROR: '{SEED_CELL_TYPE}' not found. Available: {available}")
    sys.exit(1)

pos = adata.obsm['spatial']
n = len(pos)

# build sparse inverse-distance graph 
print(f"\nBuilding sparse network (radius={RADIUS})...")
tree = cKDTree(pos)
pairs = tree.query_pairs(r=RADIUS, output_type='ndarray')
print(f"  Pairs within radius: {len(pairs):,}")

rows, cols = pairs[:, 0], pairs[:, 1]
dists = np.linalg.norm(pos[rows] - pos[cols], axis=1)

ID_sparse = csr_matrix((1.0 / dists, (rows, cols)), shape=(n, n))
ID_sparse = ID_sparse + ID_sparse.T
g = nx.from_scipy_sparse_array(ID_sparse)
print(f"  Edges before disparity filter: {g.number_of_edges():,}")

# disparity filter + alpha cut 
print("\nApplying disparity filter.../")
g = disparity_filter(g)
g = disparity_filter_alpha_cut(g, alpha_t=ALPHA)
print(f"  Edges after alpha cut (alpha={ALPHA}): {g.number_of_edges():,}")

if g.number_of_edges() == 0:
    raise RuntimeError("Disparity backbone is empty - try a larger alpha.")

# relabel integer node IDs to cell barcodes 
cell_ids = adata.obs_names.tolist()
g = nx.relabel_nodes(g, {i: cell_ids[i] for i in range(len(cell_ids))})

# component sanity check 
components = list(nx.connected_components(g))
largest = max(components, key=len)
print(f"\nComponents: {len(components):,}")
print(f"  Largest component: {len(largest):,} / {g.number_of_nodes():,} nodes "
      f"({100 * len(largest) / g.number_of_nodes():.1f}%)")

# write edge list 
os.makedirs(f"{OUT_DIR}/network", exist_ok=True)

edges_df = pd.DataFrame(
    [(u, v, d['weight']) for u, v, d in g.edges(data=True)],
    columns=['source', 'target', 'weight']
)
edges_df.to_csv(f"{OUT_DIR}/network/cells.tsv",
                sep='\t', index=False, header=False)
print(f"\nWrote {len(edges_df):,} edges -> {OUT_DIR}/network/cells.tsv")

# filter to cells that survived are in the largest CC 
seed_ids_all = adata.obs_names[adata.obs['cell_type'] == SEED_CELL_TYPE].tolist()
seed_ids     = [s for s in seed_ids_all if s in largest]
print(f"Seeds: {len(seed_ids):,} / {len(seed_ids_all):,} in largest component")

if not seed_ids:
    raise RuntimeError(f"No '{SEED_CELL_TYPE}' in largest component.")

with open(f"{OUT_DIR}/seeds.txt", 'w') as f:
    f.write('\n'.join(seed_ids))

config_yaml = f"""seed: seeds.txt

r: {RWR_RESTART}

multiplex:
  1:
    layers:
      - network/cells.tsv
    graph_type:
      - "01"
    delta: 0
    tau:
      - 1.0

eta:
  - 1.0
"""
with open(f"{OUT_DIR}/config.yml", 'w') as f:
    f.write(config_yaml)
print(f"  Wrote {OUT_DIR}/config.yml")

print(f"\nDone. Run MultiXrank with:")
print(f"  mxr = multixrank.Multixrank(config='config.yml', wdir='{OUT_DIR}')")
print(f"  ranking_df = mxr.random_walk_rank()")