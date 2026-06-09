#!/usr/bin/env python3
"""
update_seeds.py
===============
Reuses the disparity graph edge list from rwr_runs/malignant_cell and
generates seeds.txt + config.yml for a new seed cell type at a given r.

Output goes to rwr_runs/<sender_label>_<r_tag>/.

Usage: python update_seeds.py "T cell"
"""
import sys, os, json, shutil
import numpy as np
import pandas as pd
import scanpy as sc
from shapely.geometry import Point, Polygon as ShapelyPolygon

# ── parameters ────────────────────────────────────────────────────────
RWR_RESTART = 0.7        # CHANGE PER R VALUE — keep in sync with run_rwr.py & lr_score.py
R_TAG       = f"r{int(RWR_RESTART * 1000):04d}"

# ── paths ─────────────────────────────────────────────────────────────
BASE_DIR   = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH   = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"

# Graph edges built once at any r value (r-independent)
FIRST_RUN_EDGES = os.path.join(BASE_DIR, "rwr_runs/malignant_cell/network/cells.tsv")


def main():
    SEED_CT    = sys.argv[1] if len(sys.argv) > 1 else "Malignant cell"
    SEED_LABEL = SEED_CT.lower().replace(' ', '_').replace('/', '_')
    RUN_DIR    = os.path.join(BASE_DIR, f"rwr_runs/{SEED_LABEL}_{R_TAG}")
    os.makedirs(f"{RUN_DIR}/network", exist_ok=True)

    print(f"=== Updating seeds for {SEED_CT} at r={RWR_RESTART} ({R_TAG}) ===")
    print(f"Output dir: {RUN_DIR}")

    if not os.path.exists(FIRST_RUN_EDGES):
        print(f"ERROR: graph edges not found at {FIRST_RUN_EDGES}")
        print("Run export.py first to build the graph.")
        sys.exit(1)

    shutil.copy(FIRST_RUN_EDGES, f"{RUN_DIR}/network/cells.tsv")
    print(f"  Copied edge list from {FIRST_RUN_EDGES}")

    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    polygon = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([polygon.contains(Point(x, y))
                     for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()

    edges = pd.read_csv(FIRST_RUN_EDGES, sep='\t', header=None,
                        names=['source', 'target', 'weight'])
    all_nodes = set(edges['source'].astype(str)) | set(edges['target'].astype(str))

    available = adata.obs['cell_type'].unique().tolist()
    if SEED_CT not in available:
        print(f"ERROR: '{SEED_CT}' not in data. Available: {available}")
        sys.exit(1)

    seed_ids_all = adata.obs_names[adata.obs['cell_type'] == SEED_CT].tolist()
    seed_ids     = [s for s in seed_ids_all if str(s) in all_nodes]
    print(f"  Seeds: {len(seed_ids):,} / {len(seed_ids_all):,} in graph")

    if not seed_ids:
        print(f"  WARNING: no '{SEED_CT}' cells in graph — skipping")
        sys.exit(1)

    with open(f"{RUN_DIR}/seeds.txt", 'w') as f:
        f.write('\n'.join(str(s) for s in seed_ids))

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
    with open(f"{RUN_DIR}/config.yml", 'w') as f:
        f.write(config_yaml)

    print(f"  Wrote seeds.txt and config.yml -> {RUN_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()