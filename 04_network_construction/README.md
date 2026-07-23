# Network construction

Construction and comparison of spatial graph representations of the tissue: k-nearest-neighbour, Delaunay triangulation, fixed-radius, and disparity-filtered graphs.

## Contents

- **`scripts/graph.py`** - implements all four graph construction methods (`network()` function: `knn`, `eco`, `delaunay`, `disparity`), plus the disparity filter used to produce the final backbone graph. `05_random_walks/` consumes its cached output (`results/disparity_graph.pkl`) rather than re-importing this script.
- **`scripts/network_analysis.py`** - computes graph-level statistics (degree distribution, clustering coefficient, betweenness centrality).
