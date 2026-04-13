##### SpatialDM #####

import os
import pandas as pd
import numpy as np
import anndata as ad
import spatialdata as sd
import spatialdm as sdm
import spatialdm.plottings as pl
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon
from SparseAEH import MixedGaussian

original_read_csv = pd.read_csv
db_path = "/scratch/users/k22026807/masters/project/benchmarking/spatialdm/"

# figshare
if not hasattr(pd, '_is_patched'):
    original_read_csv = pd.read_csv
    pd._is_patched = True

    def patched_read_csv(filepath_or_buffer, **kwargs):
        url_mapping = {
            'https://figshare.com/ndownloader/files/36638943': os.path.join(db_path, 'interaction_input_CellChatDB.csv'),
            'https://figshare.com/ndownloader/files/36638940': os.path.join(db_path, 'complexes_input_CellChatDB.csv'),
            'https://figshare.com/ndownloader/files/36638937': os.path.join(db_path, 'cofactors.csv')
        }
        if filepath_or_buffer in url_mapping:
            return original_read_csv(url_mapping[filepath_or_buffer], **kwargs)
        return original_read_csv(filepath_or_buffer, **kwargs)
    pd.read_csv = patched_read_csv

# Load spatial and expression data
print("loading data...")
adata = ad.read_h5ad("/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad")
adata.raw = adata

# Subset to region of interest
with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[roi_mask].copy()
print(f"ROI cells: {adata.shape[0]}")


# Extract ligand receptor pairs from CellChatDB
print("extracting LR...")
sdm.extract_lr(adata, 'human', min_cell=3)

# Spatial weight matrix by rbf kernel
# l=200 for paracrine signalling
# sc = true to remove self interaction
print("calculating weight matrix...")
sdm.weight_matrix(adata, l=200, cutoff=0.2, single_cell=True)

# visualise range of interactions
plt.figure(figsize=(6, 6), facecolor='white')
weights = adata.obsp['weight'][50].toarray().flatten()
weights_log = np.log1p(weights)
plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
    c=weights_log, cmap='viridis', s=0.3, alpha=0.5, rasterized=True)
plt.colorbar(label='log(weight + 1)')
plt.title('Spatial weight distribution (cell 50)')
plt.xlabel('X (um)')
plt.ylabel('Y (um)')
plt.gca().set_facecolor('white')
plt.savefig('figures/interaction_range.png', dpi=200, facecolor='white', bbox_inches='tight')
plt.close()

# Compute Moran's R to select significant LR pairs (FDR < 0.1)
print("running spatialdm global...")
sdm.spatialdm_global(adata, method='z-score', nproc=20)
sdm.sig_pairs(adata, method='z-score', fdr=True, threshold=0.1)

# log hits to .out file
hits = adata.uns['global_res'].sort_values('z', ascending=False).index.tolist()

# Local interaction detection (z-score)
print("running spatialdm local...")
sdm.spatialdm_local(adata, method='z-score', nproc=20)
sdm.sig_spots(adata, method='z-score', fdr=False, threshold=0.1)

# Save results
adata.uns['global_res'].to_csv('figures/global_lr_results.csv')
adata.uns['local_stat']['n_spots'].to_csv('figures/local_nspots.csv')
print(f"Significant global pairs: {adata.uns['global_res'].shape[0]}")
print(f"Hits: {hits[:10]}")

# Plot discovered pairs 
#print("generating plots...")
#pl.plot_pairs(adata, hits[:3], marker='s')
#plt.savefig('/scratch/users/k22026807/masters/project/benchmarking/spatialdm/figures/discovered_pairs.png')
#plt.close()

# Manual plotting of discovered pairs as pl.plot_pairs is not functional
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
for i, pair in enumerate(hits[:3]):
    ax = axes[i]
    ax.set_facecolor('white')
    pair_idx = list(adata.uns['global_res'].index).index(pair)
    scores = np.array(adata.uns['local_stat']['local_I'][:, pair_idx]).flatten()
    pos = adata.obsm['spatial']
    ax.scatter(pos[:, 0], pos[:, 1], c='lightgrey', s=0.2, alpha=0.3, rasterized=True)
    mask = scores > 0
    if mask.sum() > 0:
        sc = ax.scatter(pos[mask, 0], pos[mask, 1], c=scores[mask], 
                       cmap='Reds', s=1, alpha=0.8, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(pair, fontsize=8)
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('figures/discovered_pairs.png', dpi=200, facecolor='white')
plt.close()

# Spatial clustering  
# Filtering for pairs with enough interacting spots
print("running spatial clustering")
lr_idx = adata.uns['local_stat']['n_spots'] > 2
bin_spots = adata.uns['selected_spots'].astype(int)[lr_idx]
Y = bin_spots.to_numpy()


### CLUSTER COMMUNICATION PATTERNS ###

try:
    print("clustering communication patterns...") 
    idx = np.random.choice(adata.shape[0], 5000, replace=False)
    gaussian_patterns = MixedGaussian(adata.obsm['spatial'][idx], group_size=16)
    gaussian_patterns.run_cluster(Y[:, idx], 5)  

    for i in range(gaussian_patterns.K):#
        plt.figure()
        plt.scatter(adata.obsm['spatial'][idx,0], adata.obsm['spatial'][idx, 1],
            c=gaussian_patterns.responsibilities[:, i], s=0.1, cmap='Reds')
        plt.savefig(f'figures/pathway_pattern_{i}.png')
        plt.close()

    # CLUSTER CELLS (NICHES)
    print("clustering cell niches...")
    gaussian_niches = MixedGaussian(adata.obsm['spatial'][idx], group_size=16)
    gaussian_niches.run_cluster(Y.T[idx, :], 5) #cluster rows (cells)

    plt.figure()
    plt.scatter(adata.obsm['spatial'][idx,0], adata.obsm['spatial'][idx,1],
        c=gaussian_niches.labels, s=0.1, cmap='plasma')
    plt.savefig('figures/cell_niches.png')
    plt.close()

    print(f"Number of cells in spatial coordinates: {adata.obsm['spatial'].shape[0]}")
    print(f"Niche labels: {len(gaussian_niches.labels)}")
    print(f"Pattern labels: {len(gaussian_patterns.labels)}")

    # PATHWAY ENRICHMENT (based on patterns)
    print("computing pathway enrichment...")
    dic = {}
    for i in range(gaussian_patterns.K):
    	dic[f'Pattern_{i}'] = bin_spots.index[gaussian_patterns.labels == i]

    pathway_res = sdm.compute_pathway(adata, dic=dic)
    pathway_res.to_csv('figures/pathway_enrichment_results.csv')

    # save pathway dot plot
    pl.dot_path(pathway_res, figsize=(6,12))
    plt.savefig('figures/pathway_dotplot.png')
    plt.close()

except Exception as e:
    print(f"SparseAEH clustering failed: {e}")  

print("saving final anndata object")

def clean_uns_for_h5ad(uns_dict):
    for key in uns_dict:
        val = uns_dict[key]
        if isinstance(val, pd.DataFrame):
            for col in val.columns:
                if val[col].dtype == object:
                    uns_dict[key][col] = val[col].astype(str)
        elif isinstance(val, pd.Series):
            uns_dict[key] = val.values 
        elif isinstance(val, dict):
            clean_uns_for_h5ad(val)

clean_uns_for_h5ad(adata.uns)
adata.write_h5ad('BC_prime_spatialdm.h5ad')

# Plotting of spatial distributions of local interactions
local_I = adata.uns['local_stat']['local_I']
global_res = adata.uns['global_res']
pos = adata.obsm['spatial']

print("local_I type:", type(local_I))
print("local_I shape:", local_I.shape if hasattr(local_I, 'shape') else 'N/A')
if hasattr(local_I, 'dtype'):
    print("local_I dtype:", local_I.dtype)
    if local_I.dtype.names:
        print("field names[:5]:", local_I.dtype.names[:5])
if hasattr(local_I, 'columns'):
    print("columns[:5]:", list(local_I.columns[:5]))

sig = global_res[global_res['selected'] == True].sort_values('z', ascending=False)
sig_dedup = sig.drop_duplicates(subset=['Ligand0']).head(6)
top_pairs = sig_dedup.index.tolist()
print("Plotting pairs:", top_pairs)

fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
axes = axes.flatten()

for i, pair in enumerate(top_pairs):
    ax = axes[i]
    ax.set_facecolor('white')
    try:
        if hasattr(local_I, 'columns'):
            if pair in local_I.columns:
                scores = np.array(local_I[pair]).flatten()
            else:
                print(f"Pair {pair} not in local_I columns, skipping")
                continue
        else:
            pair_idx = list(global_res.index).index(pair)
            scores = np.array(local_I[:, pair_idx]).flatten()
    except Exception as e:
        print(f"Could not get scores for {pair}: {e}")
        continue

    ax.scatter(pos[:, 0], pos[:, 1], c='lightgrey', s=0.2, alpha=0.3, rasterized=True)
    mask = scores > 0
    if mask.sum() > 0:
        sc = ax.scatter(pos[mask, 0], pos[mask, 1],
                        c=scores[mask], cmap='Reds', s=0.5,
                        alpha=0.8, rasterized=True,
                        vmin=0, vmax=np.percentile(scores[mask], 95))
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Local Moran's I")
    ax.set_title(pair, fontsize=9)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_aspect('equal')

plt.suptitle("Spatial Distribution of LR Interactions (SpatialDM)", fontsize=13)
plt.tight_layout()
plt.savefig('figures/local_lr_spatial.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Finished.")