import anndata as ad
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

adata = ad.read_h5ad('/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad')
pos = adata.obsm['spatial']

fig, ax = plt.subplots(figsize=(12, 18), facecolor='white')
ax.scatter(pos[:, 0], pos[:, 1], s=0.01, c='lightgrey', alpha=0.5)

colors = ['red', 'blue']
for i, name in enumerate(['region1_xenium', 'region2_xenium']):
    with open(f'/scratch/users/k22026807/masters/project/alignment/{name}.geojson') as f:
        roi = json.load(f)
    coords = roi['features'][0]['geometry']['coordinates'][0]
    poly = plt.Polygon(coords, fill=False, edgecolor=colors[i], linewidth=2, label=name)
    ax.add_patch(poly)

ax.set_aspect('equal')
ax.legend()
plt.savefig('/scratch/users/k22026807/masters/project/alignment/roi_check.png', dpi=150, bbox_inches='tight')
print('Saved roi_check.png')
