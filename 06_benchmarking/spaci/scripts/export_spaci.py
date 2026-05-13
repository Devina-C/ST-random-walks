# pulls xenium data from .h5ad file and formats for spaci R preprocessing script

import os
import anndata as ad
import pandas as pd
import numpy as np
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon
import scanpy as sc

print("Loading data...")
adata_full = ad.read_h5ad("C:/Users/Devin/Documents/st_ccc/03_celltyping/results/celltype_output/BC_prime_ROI/refined_annotations.h5ad")

# ROI selection
with open('C:/Users/Devin/Documents/st_ccc/02_preprocessing/results/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)

spatial_coords = adata_full.obsm['spatial']
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in spatial_coords])
adata = adata_full[roi_mask].copy()
adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
print(f"Extracted ROI: {adata.shape[0]} cells and {adata.shape[1]} genes.")

sc.pp.filter_cells(adata, min_counts=1)
print(f"After removing zero-count cells: {adata.shape[0]} cells")

# preprocessing
#sc.pp.normalize_total(adata, target_sum=1e4)
#sc.pp.log1p(adata)
adata.var_names = adata.var_names.str.upper()

out_dir = 'C:/Users/Devin/Documents/st_ccc/06_benchmarking/spaci/results/dataset/real_data'
os.makedirs(out_dir, exist_ok=True)

# create and export expression matrix
print("Exporting st_expression.csv...")
df_expr = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    index=adata.obs_names,
    columns=adata.var_names
).T
df_expr.index.name = None
df_expr.to_csv(os.path.join(out_dir, 'st_expression.csv'))

# create and export metadata
print("Exporting st_meta.csv...")
meta_export = pd.DataFrame({
    'x': adata.obsm['spatial'][:, 0],
    'y': adata.obsm['spatial'][:, 1],
    'type': adata.obs['cell_type'].astype(str).values
}, index=adata.obs_names)

meta_export.index.name = ''
meta_export.to_csv(os.path.join(out_dir, 'st_meta.csv'))

print("Done.")