#!/usr/bin/env python3
import os
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from graph import network, graph_properties
import networkx as nx
import matplotlib.patches as mpatches
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon

# bypass heavy calculations without editing graph.py
def skip_eff(G): 
    print("Optimization: Skipping global efficiency calculation")
    return 0.0

def skip_clos(G): 
    print("Optimization: Skipping closeness centrality calculation")
    return {}

# Overwrite the NetworkX functions globally for this session
nx.global_efficiency = skip_eff
nx.closeness_centrality = skip_clos

# Also, disable plt.show to prevent the script from hanging during drawing
import matplotlib.pyplot as plt
plt.show = lambda: None

# setup
path = "/scratch/users/k22026807/masters/project/spatial_discovery/network"
os.chdir(path)
os.makedirs('figures', exist_ok=True)

# load base AnnData
print("Loading cell type data...")
adata_ct = sc.read("../celltyping/celltype_output/BC_prime/refined_annotations.h5ad")


with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(coords)
spatial_coords = adata_ct.obsm['spatial']
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in spatial_coords])
adata_ct = adata_ct[roi_mask].copy()
print(f"ROI cells: {adata_ct.shape[0]}")
pos = adata_ct.obsm['spatial']
cell_types = adata_ct.obs['cell_type'].values

print("Calculating slide-specific action radius...")
mean_area = adata_ct.obs['cell_area'].mean()
R_mean = np.sqrt(mean_area / np.pi)

# set parameters
r = 100
alpha_val = 0.005
node_size = 0.5

# custom palette
custom_palette = {
    "Myeloid cell": "#e6550d",
    "T cell": "#5b5bd6",
    "NK cell": "#a63603",
    "B cell": "#984ea3",
    "Plasmacytoid dendritic cell": "#20b2aa",
    "Fibroblast": "#d8b365",
    "Pericyte": "#67a9cf",
    "Endothelial cell": "#66c2a5",
    "Epithelial cell": "#636363",
    "Megakaryocyte": "#fb9a99",
    "Mast cell": "#ffd92f",
    "Malignant cell": "#999999"
}

# subset positions to the cropped region for comparison
pos_crop = pos
cell_types_crop = cell_types

methods = ['knn', 'delaunay', 'radius', 'disparity']
graphs = {}

for method in methods:
    print(f"\nBuilding {method} graph...")

    if method == 'radius':
        # radius graph: connect all cells within r, no disparity filter
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        tree = cKDTree(pos_crop)
        pairs = tree.query_pairs(r=100, output_type='ndarray')
        rows, cols = pairs[:, 0], pairs[:, 1]
        dists = np.linalg.norm(pos_crop[rows] - pos_crop[cols], axis=1)
        n = len(pos_crop)
        ID_sparse = csr_matrix((1.0 / dists, (rows, cols)), shape=(n, n))
        ID_sparse = ID_sparse + ID_sparse.T
        g = nx.from_scipy_sparse_array(ID_sparse)
        pos_dict = {i: pos_crop[i] for i in range(len(pos_crop))}
        node_colors = [custom_palette.get(cell_types_crop[i], '#bdbdbd') for i in g.nodes()]
        fig, ax = plt.subplots(figsize=(12, 10))
        nx.draw(g, pos=pos_dict, with_labels=False, node_color=node_colors,
                node_size=node_size, width=0.3, edge_color='#9e9d9d', ax=ax)
        present_types = sorted(set(cell_types_crop[i] for i in g.nodes()))
        patches = [mpatches.Patch(color=custom_palette.get(ct, '#bdbdbd'), label=ct) for ct in present_types]
        ax.legend(handles=patches, loc='upper right', fontsize=6, framealpha=0.7,
                  title='Cell type', title_fontsize=7)
        ax.set_title(f'Radius graph (r={r})', fontsize=10)
        plt.savefig(f'figures/network_radius_crop.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        graphs[method] = g

    else:
        g = network(
            pos_crop,
            method=method,
            alpha=alpha_val,
            save=True,
            outdir="figures",
            cell_types=cell_types_crop,
            color_dict=custom_palette,
            node_size=node_size,
            radius=200,
            xlim=None,
            ylim=None,
            neighbors=5
        )
        os.rename(f'figures/network_{method}.png', f'figures/network_{method}_crop.png')
        graphs[method] = g


# comparison summary table
print("\n=== Graph Comparison ===")
summary = []
for method, g in graphs.items():
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    avg_deg = np.mean([d for _, d in g.degree()])
    avg_clus = nx.average_clustering(g)
    #glob_eff = nx.global_efficiency(g)
    summary.append({
        'Method': method,
        'Nodes': n_nodes,
        'Edges': n_edges,
        'Avg degree': round(avg_deg, 2),
        'Avg clustering': round(avg_clus, 4),
        'Global efficiency': "SKIPPED"
    })
    print(f"\n{method}: nodes={n_nodes}, edges={n_edges}, avg_deg={avg_deg:.2f}, "
          f"clustering={avg_clus:.4f}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv('figures/graph_comparison_summary.csv', index=False)
print("\nSummary saved to figures/graph_comparison_summary.csv")


# side-by-side comparison figure
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()
titles = {'knn': f'k-NN (k=5)', 'delaunay': 'Delaunay',
          'radius': f'Radius (r=100)', 'disparity': f'Disparity filter (α={alpha_val})'}

for ax, (method, g) in zip(axes, graphs.items()):
    pos_dict = {i: pos_crop[i] for i in range(len(pos_crop))}
    node_colors = [custom_palette.get(cell_types_crop[i], '#bdbdbd') for i in g.nodes()]
    nx.draw(g, pos=pos_dict, with_labels=False, node_color=node_colors,
            node_size=node_size, width=0.3, edge_color='#9e9d9d', ax=ax)
    n_e = g.number_of_edges()
    avg_clus = nx.average_clustering(g)
    ax.set_title(f'{titles[method]}\nEdges: {n_e} | Clustering: {avg_clus:.3f}', fontsize=11)

present_types = sorted(set(cell_types_crop))
patches = [mpatches.Patch(color=custom_palette.get(ct, '#bdbdbd'), label=ct) for ct in present_types]
fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=9,
           framealpha=0.7, title='Cell type', title_fontsize=10, bbox_to_anchor=(0.5, 0.01))

plt.suptitle('Spatial Graph Construction Method Comparison', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('figures/graph_method_comparison.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Comparison figure saved to figures/graph_method_comparison.png")


# node metrics on disparity graph (full slide)
print("\nRunning regional disparity graph and node metrics...")
graph_full = network(
    pos,
    method='disparity',
    alpha=alpha_val,
    save=True,
    outdir="figures",
    cell_types=cell_types,
    color_dict=custom_palette,
    node_size=node_size,
    radius=200,
    xlim=None,
    ylim=None,
    neighbors=5
)
os.rename('figures/network_disparity.png', f'figures/network_disparity.png')

if len(graph_full.nodes()) == 0 or graph_full.number_of_edges() == 0:
    print("Full slide backbone is empty. Skipping metrics.")
else:
    pos_dict_full = {i: pos[i] for i in range(len(pos))}
    graph_measure = graph_properties([graph_full], [pos_dict_full])
    metrics = graph_measure[0]

    node_metrics_df = pd.DataFrame(index=range(len(pos)))
    node_metrics_df['clustering_coef'] = pd.Series(metrics['clus'])
    node_metrics_df['degree_centrality'] = pd.Series(metrics['deg_cen'])
    node_metrics_df['closeness_centrality'] = pd.Series(metrics['clos_cen'])
    node_metrics_df['betweenness_centrality'] = pd.Series(metrics['bet_cen'])
    node_metrics_df['cell_type'] = cell_types
    node_metrics_df.to_csv(f'figures/node_metrics_fullslide.csv')
    print("Full slide node metrics saved.")

print("\nFinished.")
