# Cell Type Annotation

Cell type annotation using CellTypist, biomarker confirmation and neighbourhood-consensus refinement.

## Contents


- **`scripts/celltype_workflow.py`** - main annotation pipeline. Runs CellTypist against the breast cancer reference atlas, confirms low-confidence calls against curated biomarker expression, refines remaining low-confidence assignments via neighbourhood consensus in UMAP space (weighted voting among nearest neighbours) and produces the final `cell_type` annotation along with validation reports, marker expression overlays and spatial visualisations.
- **`scripts/workshop_lib.py`** - shared helper module imported by `celltype_workflow.py` (palette generation, gene-name mapping, figure saving, marker dictionary construction, spatial/matrix plotting, confidence statistics, marker-score annotation).


