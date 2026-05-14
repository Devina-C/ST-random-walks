## Unveiling cell-cell communication using random walks in spatial transcriptomics networks

This repository contains the pipeline and code used in my master's project for analysing cell-cell communication (CCC) in spatial transcriptomics data using graph-based random walk methods. 

The workflow involves spatial transcriptomics preprocessing, cell type annotation, spatial graph construction and CCC analysis. Approaches are benchmarked against alternative CCC frameworks for inference of ligand-receptor interactions. 

In addition, an exploratory resegmentation workflow was developed to assess transcript-aware whole-slide cell segmentation using Cellpose and Proseg. 

## Structure
**`resegmentation/`**: Scripts for whole-slide cell segmentation using **Cellpose** and **Proseg**, scaled using a tiling approach.

*Note: The resegmented outputs were used for methodological development not for downstream analysis.*
  
**`01_preprocessing/`**: Quality control, data normalisation and spatial data integration using the [MOSAIK](https://github.com/anthbapt/MOSAIK) workflow.

**`02_cell_typing/`**:  Workflow for marker-based cell type annotation and spatial mapping.

**`03_roi_extraction/`**: Scripts for ROI coordinate transformation; initially obtained from Xenium Explorer and converted to GeoJSON for downstream analysis.

**`04_network_construction/`**: Construction and comparison of spatial graphs using:
* disparity filtering
* radius-based
* k-nearest neighbours (kNN)
* Delaunay triangulation.

**`05_random_walks/`**: Implementation of random walk approaches to investigate cellular interactions. *Note: In progress.*

**`06_benchmarking/`**: An evaluation against established spatial analysis and  CCC methods:
* COMMOT
* NCEM
* SpaCI
* SpatialDM
* Squidpy
* stLearn

**`tools/`**: Custom utility functions.
