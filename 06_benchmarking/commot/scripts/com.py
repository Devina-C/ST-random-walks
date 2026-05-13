#### COMMOT ####
import commot as ct
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon

path = "/scratch/users/k22026807/masters/project/benchmarking/commot/"
os.chdir(path)
os.makedirs('figures', exist_ok=True)

# load anndata object
adata = ad.read_h5ad("/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad")
adata.var_names_make_unique()

# preprocessing
#sc.pp.normalize_total(adata, inplace=True)
#sc.pp.log1p(adata)

# pull LR pair database from CellChat
df_ligrec = ct.pp.ligand_receptor_database(database='CellChat', species='human')
df_ligrec.columns = ['ligand', 'receptor', 'pathway_name', 'signaling_type']
print("LR pairs:", len(df_ligrec))
print("Pathways:", df_ligrec['pathway_name'].value_counts().head(10))

# ROI selection
with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata_crop = adata[roi_mask].copy()
print(f"ROI cells: {adata_crop.shape[0]}")


# 1. CALCULATES SPATIAL DISTANCE MATRIX 
#    CHECKS EXPRESSION OF LR PAIRS
if os.path.exists('commot_result_cellchat.h5ad'):
    adata_crop = ad.read_h5ad('commot_result_cellchat.h5ad')
    print("Loaded saved CellChat COMMOT result")
else:
    ct.tl.spatial_communication(adata_crop,
        database_name='user_database',
        df_ligrec=df_ligrec,
        dis_thr=200)
    adata_crop.write_h5ad('commot_result_cellchat.h5ad')

# pulls raw matrices from spatial_communication
sender = adata_crop.obsm['commot-user_database-sum-sender']
info = adata_crop.uns['commot-user_database-info']['df_ligrec']
info.columns = ['ligand', 'receptor', 'pathway_name']

# full sender and receiver matrices
df_sender = adata_crop.obsm['commot-user_database-sum-sender'].copy()
df_receiver = adata_crop.obsm['commot-user_database-sum-receiver'].copy()

# clean up column names by stripping the 's-' and 'r-' prefixes
df_sender.columns = [c.replace('s-', '') for c in df_sender.columns]
df_receiver.columns = [c.replace('r-', '') for c in df_receiver.columns]

# attach cell barcodes
df_sender.index = adata_crop.obs_names
df_receiver.index = adata_crop.obs_names
df_sender.to_csv('figures/full_sender_matrix.csv')
df_receiver.to_csv('figures/full_receiver_matrix.csv')

# get top LR pairs by total signal
pair_totals = sender.sum(axis=0).sort_values(ascending=False)
top_pairs = pair_totals.head(10).index.tolist()
# strip 's-' prefix to get LR pair names for the later steps
top_lr = [p.replace('s-', '') for p in top_pairs]
print("Top LR pairs:", top_lr)

# save LR Pair Rankings 
pair_df = pd.DataFrame(pair_totals, columns=['Total_Signal'])
pair_df.index.name = 'LR_Pair'
pair_df.to_csv('figures/lr_pair_rankings.csv')

# Aggregate pathways (This will now work because 'sender' still has the 's-' prefixes)
pathway_totals = {}
for pathway in info['pathway_name'].dropna().unique():
    pairs = info[info['pathway_name'] == pathway]
    cols = [c for c in sender.columns if any(f's-{r["ligand"]}-{r["receptor"]}' == c for _, r in pairs.iterrows())]
    if cols:
        pathway_totals[pathway] = sender[cols].values.sum()

# top pathway selection
top_pathways = sorted(pathway_totals, key=pathway_totals.get, reverse=True)[:10]
print("Top pathways:", top_pathways)

# save Pathway Rankings
pathway_df = pd.DataFrame(list(pathway_totals.items()), columns=['Pathway', 'Total_Signal'])
pathway_df = pathway_df.sort_values('Total_Signal', ascending=False)
pathway_df.to_csv('figures/pathway_rankings.csv', index=False)

# 2. COMPUTE DIRECTION PER LR PAIR
pos = adata_crop.obsm['spatial']
for lr in top_lr:
    parts = lr.split('-', 1)
    if len(parts) == 2:
        ligand, receptor = parts
        try:
            ct.tl.communication_direction(adata_crop, database_name='user_database',
                                          lr_pair=(ligand, receptor), k=5)
        except Exception as e:
            print(f"Direction failed for {lr}: {e}")

adata_crop.write_h5ad('commot_result_cellchat.h5ad')

# vector field visualisation
vf_keys = [k for k in adata_crop.obsm.keys() if 'vf' in k and 'total' not in k]
print(f"Pathway direction keys: {vf_keys}")

for key in vf_keys:
    try:
        # extract LR pair name
        is_sender = 'sender' in key
        title_prefix = "Sender" if is_sender else "Receiver"
        lr_pair = key.split('user_database-')[-1]
        ligand, receptor = lr_pair.split('-')
        target_gene = ligand if is_sender else receptor

        # extract expression data
        if target_gene in adata_crop.var_names:
            expr = adata_crop[:, target_gene].X
            expr = expr.toarray().flatten() if hasattr(expr, "toarray") else np.array(expr).flatten()
        else:
            expr = np.zeros(len(pos))

        vectors = np.array(adata_crop.obsm[key])
        
        # GRID SETUP
        grid_size = 35
        x_grid = np.linspace(pos[:, 0].min(), pos[:, 0].max(), grid_size)
        y_grid = np.linspace(pos[:, 1].min(), pos[:, 1].max(), grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # interpolate the vector field to the grid
        u = griddata(pos, vectors[:, 0], (xx, yy), method='linear', fill_value=0.0)
        v = griddata(pos, vectors[:, 1], (xx, yy), method='linear', fill_value=0.0)
        magnitude = np.sqrt(u**2 + v**2)

        # LIGHT FILTERING
        # Hide the bottom 5% of noise
        threshold = np.max(magnitude) * 0.05
        u[magnitude < threshold] = np.nan
        v[magnitude < threshold] = np.nan

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # EXPRESSION HEATMAP BACKGROUND
        bg_cmap = 'Reds' if is_sender else 'Blues'
        scatter = ax.scatter(pos[:, 0], pos[:, 1], s=1.5, c=expr, cmap=bg_cmap, 
                             alpha=0.3, rasterized=True)

        cbar_expr = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
        cbar_expr.set_label(f'{target_gene} Expression', fontsize=9)

        # BASIC ARROWS (QUIVER)
        ax.quiver(xx, yy, u, v, 
                  color='steelblue',   # Matches the color of the original arrows
                  alpha=0.9,           # Slightly transparent so you can see expression underneath
                  angles='xy', 
                  width=0.003,         # Thickness of the arrow shaft
                  headwidth=4)         # Size of the arrow head

        ax.set_title(f"{title_prefix} Direction: {lr_pair}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'figures/commot_direction_{key}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Plot failed for {key}: {e}")

# sender/receiver maps
sender_arr = np.array(sender)
receiver_arr = np.array(adata_crop.obsm['commot-user_database-sum-receiver'])
total_sender = sender_arr.sum(axis=1)
total_receiver = receiver_arr.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
for ax, vals, title, cmap in zip(axes,
    [total_sender, total_receiver],
    ['Total Signalling Sent', 'Total Signalling Received'],
    ['Reds', 'Blues']):
    ax.set_facecolor('white')
    vmax = np.percentile(vals, 95)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap=cmap,
                    s=0.5, alpha=0.8, rasterized=True, vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(title)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('figures/commot_sender_receiver.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# cell-level hub metrics
cell_metrics = pd.DataFrame({
    'Total_Sent': total_sender,
    'Total_Received': total_receiver,
    'Cell_Type': adata_crop.obs['cell_type'].values
}, index=adata_crop.obs_names)

cell_metrics.index.name = 'Cell_Barcode'
cell_metrics.to_csv('figures/cell_level_signaling_metrics.csv')


# COMPUTE CLUSTER COMMUNICATION
print("Aggregating cell-cell communication to cluster level...")

# 1. Capture the keys BEFORE we start
keys_before = set(adata_crop.uns.keys())

for lr in top_lr:
    parts = lr.split('-', 1)
    if len(parts) == 2:
        ligand, receptor = parts
        try:
            ct.tl.cluster_communication(
                adata_crop,
                database_name='user_database',
                lr_pair=(ligand, receptor),
                clustering='cell_type',
                n_permutations=100
            )
        except Exception as e:
            print(f"Cluster computation failed for {lr}: {e}")

# 2. Identify the new keys created during the loop
keys_after = set(adata_crop.uns.keys())
cluster_result_keys = list(keys_after - keys_before)
print(f"Detected cluster result keys: {cluster_result_keys}")

# --- PLOTTING ---
if cluster_result_keys:
    # 1. FIX COORDINATES (Required for Network Plot)
    print("Calculating cluster coordinates...")
    ct.tl.cluster_position(adata_crop, clustering='cell_type')

    # 2. PLOT CLUSTER DOTPLOT
    print("Attempting Dotplot...")
    try:
        # We'll try the most basic call for older versions
        ct.pl.plot_cluster_communication_dotplot(
            adata_crop,
            clustering='cell_type',
            filename='figures/commot_dotplot.png'
        )
        print("Dotplot saved.")
    except Exception as e:
        print(f"Dotplot failed again: {e}. You can still find the raw data in lr_pair_rankings.csv")

    # 3. PLOT CLUSTER NETWORK (For the top pair)
    print("Attempting Network Plot...")
    try:
        # Using the first detected key from your log
        target_key = cluster_result_keys[0] 
        ct.pl.plot_cluster_communication_network(
            adata_crop,
            uns_names=[target_key], # Some versions use uns_names as a list
            clustering='cell_type',
            filename='figures/commot_network_top.png'
        )
        print("Network plot saved.")
    except Exception as e:
        print(f"Network plot failed: {e}")

print("Finished.")