#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.spatial.distance as dist
import sklearn.neighbors as skn
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.spatial as spatial
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def palette(key = 0):
    """
    Returns a color palette based on the specified key.

    Args:
        key (int, optional): Determines the palette to return. If `key` is 0, returns a 
            custom RGBA color palette. Otherwise, returns a dictionary of standard color 
            names mapped to integer keys. Default is 0.

    Returns:
        dict: A dictionary where each key corresponds to a color. If `key` is 0, 
            the values are RGBA tuples representing specific custom colors:
            - 0: (0.604, 0.678, 0.749, 1)  # '#9AADBF'
            - 1: (0.827, 0.725, 0.624, 1)  # '#D3B99F'
            - 2: (0.757, 0.467, 0.404, 1)  # '#C17767'
            - 3: (0.427, 0.596, 0.729, 1)  # '#6D98BA'
            
            If `key` is not 0, the values are standard color names as strings:
            - 0: 'red'
            - 1: 'blue'
            - 2: 'green'
            - 3: 'orange'
            - 4: 'gold'
            - ...
            - 29: 'khaki'
    """
    
    if key == 0:
        color1 = (0.604, 0.678, 0.749, 1) # '#9AADBF'
        color2 = (0.827, 0.725, 0.624, 1) # '#D3B99F'
        color3 = (0.757, 0.467, 0.404, 1) # '#C17767'
        color4 = (0.427, 0.596, 0.729, 1) # '#6D98BA'
        colors = {0:color1 , 1:color2, 2:color3, 3:color4}
        
        return colors
    
    else:    
        add_colors = {0:'teal', 1:'orange', 2:'green', 3:'red', 4:'gold', 5:'purple', 6:'grey', 7:'pink',\
                 8:'navy', 9:'springgreen', 10:'salmon', 11:'skyblue', 12:'tan', 13:'sienna',\
                 14:'turquoise', 15:'aqua', 16:'chartreuse', 17:'crimson', 18:'fuchsia', 19:'beige',\
                 20:'yellow', 21:'blue', 22:'olivedrab', 23:'deeppink', 24:'maroon', 25:'mistyrose',\
                 26:'seagreen', 27:'darkorange', 28:'mediumpurple', 29:'khaki'}
            
        return add_colors

colors = palette(1)

def disparity_filter(G, weight = 'weight'):
    """
    Compute significance scores (alpha) for weighted edges in a graph using the disparity filter method 
    as defined by Serrano et al. (2009).

    Args:
        G (networkx.Graph or networkx.DiGraph): 
            A weighted graph (either undirected or directed) for which the significance scores will be computed.
        weight (str, optional): 
            The edge attribute that specifies the weight of each edge. Default is 'weight'.

    Returns:
        networkx.Graph or networkx.DiGraph: 
            A graph with the same structure as `G`, but with an additional alpha score assigned to each edge.
            - For directed graphs:
                - `alpha_out` is added for edges based on outgoing connections.
                - `alpha_in` is added for edges based on incoming connections.
            - For undirected graphs:
                - `alpha` is added for each edge.
        
    Raises:
        ValueError: If the input graph does not have the specified `weight` attribute for its edges.
        TypeError: If `G` is not a NetworkX graph or digraph.

    References:
        Serrano, M. A., Boguñá, M., & Vespignani, A. (2009). Extracting the Multiscale Backbone of Complex
        Weighted Networks. Proceedings of the National Academy of Sciences, 106(16), 6483-6488.
        https://doi.org/10.1073/pnas.0808904106
    """
    
    if nx.is_directed(G): #directed case    
        N = nx.DiGraph()
        for u in G:
            
            k_out = G.out_degree(u)
            k_in = G.in_degree(u)
            
            if k_out > 1:
                sum_w_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
                for v in G.successors(u):
                    w = G[u][v][weight]
                    p_ij_out = float(np.absolute(w))/sum_w_out
                    alpha_ij_out = 1 - (k_out-1) * integrate.quad(lambda x: (1-x)**(k_out-2), 0, p_ij_out)[0]
                    N.add_edge(u, v, weight = w, alpha_out=float('%.4f' % alpha_ij_out))
                    
            elif k_out == 1 and G.in_degree(G.successors(u)[0]) == 1:
                #we need to keep the connection as it is the only way to maintain the connectivity of the network
                v = G.successors(u)[0]
                w = G[u][v][weight]
                N.add_edge(u, v, weight = w, alpha_out = 0, alpha_in = 0)
                #there is no need to do the same for the k_in, since the link is built already from the tail
            
            if k_in > 1:
                sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
                for v in G.predecessors(u):
                    w = G[v][u][weight]
                    p_ij_in = float(np.absolute(w))/sum_w_in
                    alpha_ij_in = 1 - (k_in-1) * integrate.quad(lambda x: (1-x)**(k_in-2), 0, p_ij_in)[0]
                    N.add_edge(v, u, weight = w, alpha_in=float('%.4f' % alpha_ij_in))
        return N
    
    else: #undirected case
        B = nx.Graph()
        for u in G:
            k = len(G[u])
            if k > 1:
                sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
                for v in G[u]:
                    w = G[u][v][weight]
                    p_ij = float(np.absolute(w))/sum_w
                    alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                    B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
                    
        return B


def disparity_filter_alpha_cut(G, weight = 'weight', alpha_t = 0.3, cut_mode = 'or'):
    """
    Filter edges from a graph based on the alpha significance scores generated by the `disparity_filter` function.

    Args:
        G (networkx.Graph or networkx.DiGraph): 
            The input graph previously filtered using the `disparity_filter` function.
        weight (str, optional): 
            The edge attribute key specifying edge weights. Default is 'weight'.
        alpha_t (float, optional): 
            Threshold for the alpha parameter. Edges with alpha scores greater than or equal to this value 
            will be removed. Must be a value between 0 and 1. Default is 0.4.
        cut_mode (str, optional): 
            Specifies the logical operation for filtering edges in directed graphs:
            - `'or'`: Keeps edges if either `alpha_in` or `alpha_out` is below the threshold.
            - `'and'`: Keeps edges only if both `alpha_in` and `alpha_out` are below the threshold.
            Default is `'or'`.

    Returns:
        networkx.Graph or networkx.DiGraph: 
            A new graph containing only the edges that pass the alpha threshold filtering.
            - For directed graphs:
                - The `cut_mode` determines the filtering logic.
            - For undirected graphs:
                - Edges are retained if their `alpha` score is below the threshold.

    Raises:
        ValueError: If `alpha_t` is not within the range [0, 1].
        TypeError: If `G` is not a NetworkX graph or digraph.
        KeyError: If the input graph's edges do not contain the required `alpha` attributes.
    """   
    
    if nx.is_directed(G): # Directed case:   
        B = nx.DiGraph()
        for u, v, w in G.edges(data = True):
            try:
                alpha_in =  w['alpha_in']
            except KeyError: #there is no alpha_in, so we assign 1. It will never pass the cut
                alpha_in = 1
            try:
                alpha_out =  w['alpha_out']
            except KeyError: #there is no alpha_out, so we assign 1. It will never pass the cut
                alpha_out = 1  
            
            if cut_mode == 'or':
                if alpha_in < alpha_t or alpha_out<alpha_t:
                    B.add_edge(u,v, weight = w[weight])
            elif cut_mode == 'and':
                if alpha_in < alpha_t and alpha_out<alpha_t:
                    B.add_edge(u,v, weight = w[weight])
        return B

    else: # Undirected case  
        B = nx.Graph() 
        for u, v, w in G.edges(data = True):
            
            try:
                alpha = w['alpha']
            except KeyError: #there is no alpha, so we assign 1. It will never pass the cut
                alpha = 1
                
            if alpha < alpha_t:
                B.add_edge(u,v, weight = w[weight])
                
        return B                


def eco(W, directed = False):
    """
    Filter a weighted square similarity matrix to obtain a binary adjacency matrix based on a topological criterion.

    Args:
        W (numpy.ndarray): 
            A square matrix representing weighted similarities between nodes. Must be symmetric if the network is undirected.
        directed (bool, optional): 
            Indicates whether the network is directed (`True`) or undirected (`False`). Default is `False`.

    Returns:
        numpy.ndarray: 
            A binary adjacency matrix where edges represent significant connections based on the filtering criterion.

    Raises:
        ValueError: If the input matrix is too sparse to form the required number of connections (`3*N` for directed 
        or `1.5*N` for undirected graphs).

    References:
        De Vico Fallani, F., Latora, V., & Chavez, M. (2017). A topological criterion to filter information 
        in complex brain networks. *PLOS Computational Biology*, 13(1), e1005305.
        https://doi.org/10.1371/journal.pcbi.1005305
    """

    N = W.shape[0]
    if directed:
        numcon = 3 * N
        ind = np.nonzero(W)
    else:
        W = np.triu(W)
        numcon = int(1.5 * N)
        ind = np.nonzero(np.triu(W))

    if numcon > len(ind[0]):
        raise ValueError("Input matrix is too sparse")

    weights = W[ind]
    sorind = sorted(zip(ind[0], ind[1], weights), key=lambda x: -x[2])

    # Zero out the weights of elements after the `numcon` threshold
    for i in range(numcon, len(sorind)):
        W[sorind[i][0], sorind[i][1]] = 0

    # Form the adjacency matrix
    if directed:
        A = (W != 0).astype(int)
    else:
        A = (W + W.T != 0).astype(int)

    return A
 

def inverse_distance(D):
    """
    Computes the inverse of the distance matrix, setting entries corresponding to zero distances to zero.

    Args:
        D (numpy.ndarray): 
            A 2D square matrix of distances, where `D[i, j]` represents the distance between point `i` and point `j`. 
            The matrix should be of shape `(n, n)` where `n` is the number of points.

    Returns:
        numpy.ndarray: 
            A 2D square matrix of the same shape as `D`, where each entry `ID[i, j]` is the inverse of `D[i, j]`
            if `D[i, j]` is non-zero, or `0` if `D[i, j]` is zero.
    """
    
    ID = np.zeros(D.shape)
        
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i,j] != 0:
                ID[i,j] = 1/D[i,j]
    return ID
            

def network(points, method='knn', neighbors=3, alpha=0.25, save=True, outdir=None, node_size=3,
            cell_types=None, color_dict=None, radius=None, xlim=None, ylim=None):
    
    n = len(points)
    
    """
    Constructs a graph from segmented nuclei centroids using specified methods.

    Args:
        points (numpy.ndarray): 
            Array of 2D or 3D coordinates representing the nuclei centroids.
        method (str, optional): 
            Graph construction method. Options are:
            - `'knn'`: Constructs a k-nearest neighbor (k-NN) graph.
            - `'eco'`: Constructs a graph using the ECO filtering method.
            - `'disparity'`: Constructs a graph filtered using the disparity filter and alpha cut.
            Default is `'knn'`.
        neighbors (int, optional): 
            Number of neighbors for the k-NN graph when `method='knn'`. Default is `3`.
        alpha (float, optional):
            Alpha threshold for the disparity filter cut. Default is `0.25`.
        save (bool, optional): 
            Whether to save a visualization of the constructed graph as a `.png` file. Default is `True`.
        outdir (str, optional):
            Directory in which to save the output figure. Default is `None` (current directory).
        cell_types (array-like, optional):
            Sequence of cell-type labels, one per node (same order as `points`).
            When provided, nodes are coloured by cell type using `color_dict`.
            If `None`, all nodes are drawn in a single default colour.
        color_dict (dict, optional):
            Mapping from cell-type label → hex/RGB colour string.
            Used only when `cell_types` is provided.
            If `None` but `cell_types` is given, a generic matplotlib colour cycle is used.

    Returns:
        networkx.Graph: 
            The constructed graph as a NetworkX graph object.

    Raises:
        ValueError: 
            If an unsupported `method` is provided.
    """

    # ── build the graph ────────────────────────────────────────────────────
    if method == 'knn':
        A = skn.kneighbors_graph(points, n_neighbors=neighbors)
        graph = nx.from_numpy_array(A)

    elif method == 'eco':
        condenced_D = dist.pdist(points, 'euclidean')
        D = dist.squareform(condenced_D, force='no', checks=True)
        ID = inverse_distance(D)
        A = eco(ID)
        graph = nx.from_numpy_array(A)

    elif method == 'delaunay':
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = simplex[i], simplex[j]
                    edges.add((min(u,v), max(u,v)))
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for u, v in edges:
            d = np.linalg.norm(points[u] - points[v])
            if d > 0:
                graph.add_edge(u, v, weight=1.0/d)

    elif method == 'disparity':
        if radius is not None:
            print(f"Build sparse network with cKDTree (radius{radius:.2f})...")
            from scipy.spatial import cKDTree
            from scipy.sparse import csr_matrix

            tree = cKDTree(points)
            pairs = tree.query_pairs(r=radius, output_type='ndarray')

            rows, cols = pairs[:, 0], pairs[:, 1]
            dists = np.linalg.norm(points[rows] - points[cols], axis=1)
                        
            # fast sparse inverse distance
            ID_sparse = csr_matrix((1.0 / dists, (rows, cols)), shape=(n, n))
            ID_sparse = ID_sparse + ID_sparse.T
            graph = nx.from_scipy_sparse_array(ID_sparse)
            
            print(f"Edges in backbone: {graph.number_of_edges()}")

            print("Applying disparity filter...")
            graph = disparity_filter(graph)

            print(f"Edges after disparity filter: {graph.number_of_edges()}")
            graph = disparity_filter_alpha_cut(graph, alpha_t=alpha)

            print(f"Edges after alpha cut ({alpha}): {graph.number_of_edges()}")
        else:
            #OLD DENSE METHOD
            print("WARNING: Building dense distance matrix. May crash on large datasets.")
            condenced_D = dist.pdist(points, 'euclidean')
            D = dist.squareform(condenced_D, force='no', checks=True)
            ID = inverse_distance(D)
            graph = nx.from_numpy_array(ID)


    else:
        raise ValueError("Please provide an implemented method among: knn, eco, disparity")

    # ── build node-colour list ─────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pos_dict = {i: points[i] for i in range(len(points))}

    if cell_types is not None:
        cell_types = list(cell_types)   # ensure indexable

        # If no colour dict supplied, build a generic one from matplotlib tab20
        if color_dict is None:
            unique_types = sorted(set(cell_types))
            cmap = plt.get_cmap('tab20', len(unique_types))
            color_dict = {ct: cmap(i) for i, ct in enumerate(unique_types)}

        node_colors = [color_dict.get(cell_types[n], '#bdbdbd') for n in graph.nodes()]

        fig, ax = plt.subplots(figsize=(12, 10))
        nx.draw(
            graph,
            pos=pos_dict,
            with_labels=False,
            node_color=node_colors,
            node_size=node_size,
            width=0.6,
            edge_color='#9e9d9d',
            ax=ax,
        )

        # ── legend ────────────────────────────────────────────────────────
        present_types = sorted(set(cell_types[n] for n in graph.nodes()))
        patches = [
            mpatches.Patch(color=color_dict.get(ct, '#bdbdbd'), label=ct)
            for ct in present_types
        ]
        ax.legend(
            handles=patches,
            loc='upper right',
            fontsize=6,
            framealpha=0.7,
            title='Cell type',
            title_fontsize=7,
            markerscale=1.2,
        )

    else:
        # ── original behaviour (no cell-type info) ─────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(
            graph,
            pos=pos_dict,
            with_labels=True,
            font_size=2,
            width=0.5,
            node_size=10,
            ax=ax,
        )
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # ── save / close ───────────────────────────────────────────────────────
    if save:
        fname = f'network_{method}.png'
        fpath = (outdir + '/' + fname) if outdir is not None else fname
        plt.savefig(fpath, format='png', dpi=600, bbox_inches='tight')

    plt.close()
    return graph


def graph_properties(list_graphs, list_pos):
    """
    Compute and visualize various properties of a list of graphs and their layouts.

    Args:
        list_graphs (list of networkx.Graph): A list of NetworkX graph objects for which 
            properties will be computed.
        list_pos (list of dict): A list of layout dictionaries for the corresponding graphs 
            in `list_graphs`. Each layout is a dictionary of node positions, typically 
            generated by a NetworkX layout function (e.g., `nx.spring_layout`).

    Returns:
        dict: A dictionary where each key is the index of a graph and the value is another 
        dictionary containing graph properties:
            - `avg_clus` (float): Average clustering coefficient.
            - `glob_effi` (float): Global efficiency of the graph.
            - `clus` (dict): Local clustering coefficients for each node.
            - `deg_cen` (dict): Degree centrality for each node.
            - `clos_cen` (dict): Closeness centrality for each node.
            - `bet_cen` (dict): Betweenness centrality for each node.
            - `eigen_cen` (dict, optional): Eigenvector centrality for each node, included 
              only if the graph is connected.

    Raises:
        ValueError: If `list_graphs` and `list_pos` do not have the same length.
    """
    
    compt = 0
    dict_graphs = dict()
    for i,j in zip(list_graphs, list_pos):
        nx.draw(i, j, with_labels = True, font_size = 2, width = 0.5, node_size = 10, node_color = colors[compt])
        plt.show()
        plt.close()
        avg_clus = nx.average_clustering(i)
        glob_effi = nx.global_efficiency(i)
        clus = nx.clustering(i)
        deg_cen = nx.degree_centrality(i)
        clos_cen = nx.closeness_centrality(i)
        num_nodes = len(i.nodes())
        k_value = min(1000, num_nodes)
        bet_cen = nx.betweenness_centrality(i, k=k_value)
        if nx.is_connected(i) == True:
            eigen_cen = nx.eigenvector_centrality(i)
            temp_dict = {'avg_clus': avg_clus, 'glob_effi': glob_effi, 'clus': clus, \
                     'deg_cen': deg_cen, 'eigen_cen': eigen_cen, 'clos_cen': clos_cen, \
                     'bet_cen': bet_cen}
        else:
            temp_dict = {'avg_clus': avg_clus, 'glob_effi': glob_effi, 'clus': clus, \
                     'deg_cen': deg_cen, 'clos_cen': clos_cen, 'bet_cen': bet_cen}
        dict_graphs[compt] = temp_dict
        compt += 1
    
    return dict_graphs

    
def degree_hist(list_graphs):
    """
    Plot and save degree histograms for a list of graphs.

    Args:
        list_graphs (list of networkx.Graph): A list of NetworkX graph objects for which 
            degree histograms will be computed and plotted.

    Returns:
        None: The function does not return a value. It generates and saves a plot of the 
        degree histograms.

    Raises:
        ValueError: If the `colors` variable used in the function is not defined or does 
        not have enough entries to cover all graphs in `list_graphs`.
    """
    
    compt = 0
    for i in list_graphs:
        degree_sequence = sorted((d for n, d in i.degree()), reverse = True)
        plt.plot(degree_sequence, color = colors[compt], marker = ".", label = str(compt))
        plt.legend()
        compt += 1
        
    plt.savefig('degree_histogram.png', format = 'png', dpi = 600)
    plt.show()