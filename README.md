## Unveiling cell-cell communication using random walks in spatial transcriptomics networks

This repository contains the pipeline and code used in my master's project. Analysis was performed on publicly available Xenium FFPE Human Breast Cancer data.

## Overview
This project explores cell-cell communication (CCC) in the tumour microenvironment using random walks. It benchmarks these methods against commonly used CCC tools which adopt differing statistical and graph-based approaches to model ligand-receptor interactions. 

## Pipeline
* **`01_preprocessing/`**: Quality control, data normalisation and spatial data integration using the [MOSAIK](https://github.com/anthbapt/MOSAIK) workflow.
* **`02_cell_typing/`**:  Workflow for marker-based cell type assignment and spatial mapping.
* **`03_roi_extraction/`**: Scripts for ROI coordinate transformation; initially obtained from Xenium Explorer and converted to GeoJSON for downstream analysis.
* **`04_network_construction/`**: Spatial graph construction using a disparity filter. Compared against other methods, such as radius-based, k-Nearest Neighbours and Delaunay Triangulation.
* **`05_random_walks/`**: Implementation of random walk approaches to investigate cellular interactions.
* **`06_benchmarking/`**: An evaluation of current CCC methods (`COMMOT`, `NCEM`, `SpaCI`, `SpatialDM`, `Squidpy`, `stLearn`).
* **`tools/`**: Custom utility functions.
* **`resegmentation/`**: Scripts for cell boundary segmentation using Cellpose and Proseg, scaled to process whole-slide images via a tiling approach. 
