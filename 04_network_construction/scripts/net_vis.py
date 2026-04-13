### NETWORK VISUALISATION ### 

#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Setup
path = "/scratch/users/k22026807/masters/project/spatial_discovery/"
os.chdir(path)
os.makedirs('figures', exist_ok=True)

# 2. Load Data
print("Loading data for coordinate check...")
adata = sc.read("../celltyping/celltype_output/BC_prime/refined_annotations.h5ad")
pos = adata.obsm['spatial']
x = pos[:, 0]
y = pos[:, 1]

# 3. Print the absolute bounds to the log
print(f"--- COORDINATE ANALYSIS ---")
print(f"X-Axis Range: {x.min():.1f} to {x.max():.1f}")
print(f"Y-Axis Range: {y.min():.1f} to {y.max():.1f}")
print(f"Total Cells: {len(pos)}")

# 4. Create a 2-Panel Diagnostic Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Panel A: Density Heatmap (Finds the clusters)
hb = ax1.hexbin(x, y, gridsize=100, cmap='YlOrRd', mincnt=1)
fig.colorbar(hb, ax=ax1, label='Cell Count per Bin')
ax1.set_title("Cell Density")
ax1.set_xlabel("X coordinate (Microns/Pixels)")
ax1.set_ylabel("Y coordinate (Microns/Pixels)")

# Panel B: Global Cell Type Map (Matches your previous JPGs but with AXES)
# We use a very small point size to see the overall shape
sc.pl.spatial(
	adata, 
	color='cell_type',
	spot_size=30, 
	alpha=1,
	ax=ax2, 
	show=False)

ax2.set_axis_on()
ax2.tick_params(labelsize=10, labelbottom=True, labelleft=True)
ax2.set_xlabel("X (Microns)")
ax2.set_ylabel("Y (Microns)")
ax2.set_title("Global Cell Type Map")

plt.tight_layout()
plt.savefig('figures/coordinate_diagnostic_map.png', dpi=300)
print("Diagnostic map saved to figures/coordinate_diagnostic_map.png")