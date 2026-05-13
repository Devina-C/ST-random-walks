#### spaCI ####
# Spatial Cellular Intercellular communication
# Random Forest with spatial regularization

import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon

try:
    import spaCI
except ImportError:
    print("spaCI not found. Please install via GitHub.")

# Set paths
path = "/scratch/users/k22026807/masters/project/benchmarking/spaci/"
os.chdir(path)
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

#========================
#     1. LOAD DATA 
#========================
print("Loading data...")
adata = ad.read_h5ad("/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad")

# ROI selection (matching your other scripts)
with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[roi_mask].copy()

# Preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# spaCI requires coordinates in a specific format in obs or obsm
adata.obs['x'] = adata.obsm['spatial'][:, 0]
adata.obs['y'] = adata.obsm['spatial'][:, 1]

#========================
#   2. spaCI ANALYSIS
#========================
# Step 1: Initialize the spaCI object
# We use the CellChat database as you did with COMMOT and SpatialDM
print("Initializing spaCI...")
spaci_obj = spaCI.spaCI(
    adata, 
    label='cell_type', 
    database='CellChat', 
    species='human'
)

# Step 2: Run the communication inference
# distance_threshold=200 matches your NCEM/COMMOT radius
print("Inferring spatial communication (this may take time)...")
spaci_obj.run_spaci(
    distance_threshold=200, 
    n_jobs=20,        # Optimized for your HPC login/interrupt nodes
    min_cells=3       # Minimum cells expressing LR pair
)

# Step 3: Save results
print("Saving results...")
df_inter = spaci_obj.get_interactions()
df_inter.to_csv('results/spaci_interactions.csv', index=False)

#========================
#    3. VISUALIZATION
#========================
# Plot 1: Top 10 Interactions Heatmap
top_10 = df_inter.head(10)
print(f"Top detected interactions:\n{top_10[['ligand', 'receptor', 'p_value']]}")

# If spaCI has built-in plotting:
try:
    spaci_obj.plot_interaction_heatmap(save='figures/spaci_heatmap.png')
except Exception as e:
    # Manual plot if built-in fails
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['ligand'] + " - " + top_10['receptor'], -np.log10(top_10['p_value']))
    plt.xlabel("-log10(p-value)")
    plt.title("Top 10 spaCI Interactions")
    plt.savefig('figures/top_interactions_bar.png')
    plt.close()

print("Finished spaCI analysis.")