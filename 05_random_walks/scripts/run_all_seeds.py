#!/usr/bin/env python3
"""
run_all_seeds.py
================
Builds disparity graph once, then loops over all seed cell types
running update_seeds → RWR → visualise for each.
"""
import subprocess
import sys
import os

BASE_DIR = "C:/Users/Devin/Documents/ST_ccc/05_random_walks"

SEED_TYPES = [
    "Malignant cell",
    "T cell",
    "Myeloid cell",
    "Fibroblast",
    "Endothelial cell",
    "B cell",
    "Pericyte",
    "Epithelial cell",
    "Plasmacytoid dendritic cell",
    "Mast cell",
]

python = sys.executable

# Step 1: build disparity graph once using first seed type
print("="*60)
print("Building disparity graph (runs once)...")
print("="*60)
#subprocess.run([python, "scripts/export.py", SEED_TYPES[0]], check=True)
subprocess.run([python, "scripts/run_rwr.py",   "malignant_cell"], check=True)
subprocess.run([python, "scripts/visualise.py", "malignant_cell", SEED_TYPES[0]], check=True)

# Step 2: loop over all seed types
for seed_ct in SEED_TYPES[1:]:
    seed_label = seed_ct.lower().replace(' ', '_').replace('/', '_')
    print(f"\n{'='*60}")
    print(f"SEED: {seed_ct}  (label: {seed_label})")
    print('='*60)

    print("→ Updating seeds and config...")
    subprocess.run([python, "scripts/update_seeds.py", seed_ct], check=True)

    print("→ Running RWR...")
    subprocess.run([python, "scripts/run_rwr.py", seed_label], check=True)

    print("→ Visualising...")
    subprocess.run([python, "scripts/visualise.py", seed_label, seed_ct],
                   check=True)

    print(f"✓ Done: {seed_label}")

print("\nAll seed types complete.")