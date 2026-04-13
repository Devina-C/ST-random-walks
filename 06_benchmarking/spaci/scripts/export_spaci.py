# pulls xenium data from .h5ad file and formats for spaci R script

import os
import anndata as ad
import pandas as pd
import numpy as np
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon
import scanpy as sc

print("Loading data...")
adata_full = ad.read_h5ad("/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad")

# ROI selection
with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)

spatial_coords = adata_full.obsm['spatial']
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in spatial_coords])
adata = adata_full[roi_mask].copy()
adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()

print(f"Extracted ROI: {adata.shape[0]} cells and {adata.shape[1]} genes.")

# preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

os.makedirs('dataset/real_data', exist_ok=True)

# create st_meta.csv
print("Exporting st_meta.csv...")
meta_df = pd.DataFrame({
    'dimx': adata.obsm['spatial'][:, 0],
    'dimy': adata.obsm['spatial'][:, 1],
    'cell_type': adata.obs['cell_type'].values
}, index=adata.obs_names)
meta_df.to_csv('dataset/real_data/st_meta.csv')

# create st_expression.csv
print("Exporting st_expression.csv...")

# convert sparse matrix to dense DataFrame, then transpose
if hasattr(adata.X, 'toarray'):
    exp_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names).T
else:
    exp_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names).T

exp_df.to_csv('dataset/real_data/st_expression.csv')

print("Success! Data is ready for spaCI_preprocess.R")