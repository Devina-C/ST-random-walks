import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ss
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import math
import os
from matplotlib.colors import ListedColormap


def get_laplacian_mtx(adata,
                      num_neighbors=6,
                      spatial_key=['array_row', 'array_col'],
                      normalization=False):
    """
    Obtain the Laplacian matrix or normalized laplacian matrix.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates could be found in adata.obs
        or adata.obsm.
    num_neighbors: int, optional
        The number of neighbors for each node/spot/pixel when construct graph.
        The default if 6.
    spatial_key=None : list | string
        Get the coordinate information by adata.obsm[spaital_key] or adata.var[spatial_key].
        The default is ['array_row', 'array_col'].
    normalization : bool, optional
        Whether you need to normalize laplacian matrix. The default is False.

    Raises
    ------
    KeyError
        The coordinates should be found at adata.obs[spatial_names] or adata.obsm[spatial_key]

    Returns
    -------
    lap_mtx : csr_matrix
        The Laplacian matrix or normalized Laplacian matrix.

    """
    if spatial_key in adata.obsm_keys():
        adj_mtx = kneighbors_graph(adata.obsm[spatial_key],
                                   n_neighbors=num_neighbors)
    elif set(spatial_key) <= set(adata.obs_keys()):
        adj_mtx = kneighbors_graph(adata.obs[spatial_key],
                                   n_neighbors=num_neighbors)
    else:
        raise KeyError("%s is not available in adata.obsm_keys" % spatial_key + " or adata.obs_keys")

    adj_mtx = nx.adjacency_matrix(nx.Graph(adj_mtx))
    # Degree matrix
    deg_mtx = adj_mtx.sum(axis=1)
    deg_mtx = create_degree_mtx(deg_mtx)
    # Laplacian matrix
    # Whether you need to normalize Laplacian matrix
    if not normalization:
        lap_mtx = deg_mtx - adj_mtx
    else:
        deg_mtx = np.array(adj_mtx.sum(axis=1)) ** (-0.5)
        deg_mtx = create_degree_mtx(deg_mtx)
        lap_mtx = ss.identity(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx

    return lap_mtx


def create_adjacent_mtx(coor_df,
                        spatial_names=['array_row', 'array_col'],
                        num_neighbors=4):
    # Transform coordinate dataframe to coordinate array
    coor_array = coor_df.loc[:, spatial_names].values
    coor_array.astype(np.float32)
    edge_list = []
    num_neighbors += 1
    for i in range(coor_array.shape[0]):
        point = coor_array[i, :]
        distances = np.sum(np.asarray((point - coor_array) ** 2), axis=1)
        distances = pd.DataFrame(distances,
                                 index=range(coor_array.shape[0]),
                                 columns=["distance"])
        distances = distances.sort_values(by='distance', ascending=True)
        neighbors = distances[1:num_neighbors].index.tolist()
        edge_list.extend((i, j) for j in neighbors)
        edge_list.extend((j, i) for j in neighbors)
    # Remove duplicates
    edge_list = set(edge_list)
    edge_list = list(edge_list)
    row_index = []
    col_index = []
    row_index.extend(j[0] for j in edge_list)
    col_index.extend(j[1] for j in edge_list)

    sparse_mtx = ss.coo_matrix((np.ones_like(row_index), (row_index, col_index)),
                               shape=(coor_array.shape[0], coor_array.shape[0]))

    return sparse_mtx


def create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)),
                               shape=(diag.size, diag.size))

    return sparse_mtx


def gene_clustering_kMeans(frequency_array, n_clusters, reduction=50):
    # Normalization
    frequency_array = preprocessing.StandardScaler().fit_transform(frequency_array)

    # Dimension reduction
    if reduction and frequency_array.shape[1] > reduction:
        pca = PCA(n_components=reduction)
        frequency_array = pca.fit_transform(frequency_array)

    # Clustering
    kmeans_model = KMeans(n_clusters=n_clusters).fit(frequency_array)

    return kmeans_model.labels_


def window_side_bin(adata,
                    shape=(20, 20),
                    spatial_names=['array_row', 'array_col'],
                    sparse=True):
    # Extract border
    max_x = adata.obs[spatial_names[1]].max()
    min_x = adata.obs[spatial_names[1]].min()
    max_y = adata.obs[spatial_names[0]].max()
    min_y = adata.obs[spatial_names[0]].min()

    # Calculate bin-size
    bin_x = (max_x - min_x) / (shape[0] - 1)
    bin_y = (max_y - min_y) / (shape[1] - 1)

    # Create a new dataframe to store new coordinates
    new_coor_df = pd.DataFrame(0, index=adata.obs.index, columns=spatial_names)
    for i in range(adata.shape[0]):
        coor_x = adata.obs.iloc[i][spatial_names][1]
        coor_y = adata.obs.iloc[i][spatial_names][0]
        coor_x = int(np.floor((coor_x - min_x) / bin_x))
        coor_y = int(np.floor((coor_y - min_y) / bin_y))
        new_coor_df.iloc[i, :] = [coor_x, coor_y]

    # Merge the spots within the bins
    count_mtx = pd.DataFrame(columns=adata.var_names)
    final_coor_df = pd.DataFrame(columns=spatial_names)
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp_index = new_coor_df[new_coor_df.iloc[:, 0] == i]
            tmp_index = tmp_index[tmp_index.iloc[:, 1] == j]
            if tmp_index.shape[0] > 0:
                tmp_index = tmp_index.index
                count_mtx.loc["pseudo_" + str(i) + "_" + str(j), :] = adata[tmp_index, :].X.sum(axis=0)
                final_coor_df.loc["pseudo_" + str(i) + "_" + str(j), :] = [i, j]
            else:
                count_mtx.loc["pseudo_" + str(i) + "_" + str(j), :] = 0
                final_coor_df.loc["pseudo_" + str(i) + "_" + str(j), :] = [i, j]

    # Transform it to anndata
    from anndata import AnnData
    if sparse == True:
        from scipy import sparse
        new_adata = AnnData(sparse.coo_matrix(count_mtx))
        new_adata.obs_names = count_mtx.index.tolist()
        new_adata.var_names = count_mtx.columns.tolist()
        new_adata.obs = final_coor_df
    else:
        new_adata = AnnData(count_mtx)
        new_adata.obs = final_coor_df

    return new_adata


def select_svg_normal(gene_score, num_sigma=1):
    mu = np.mean(gene_score['gft_score'])
    sigma = np.std(gene_score['gft_score'])
    gene_score['spatially_variable'] = 0
    gene_score.loc[gene_score['gft_score'] > mu + num_sigma * sigma,
    'spatially_variable'] = 1

    return gene_score


def select_svg_kmean(gene_score):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    X = gene_score["smooth_score"].tolist()
    X = np.array(X).reshape(-1, 1)
    y_pred = kmeans.fit_predict(X)
    gene_score['spatially_variable'] = y_pred

    return gene_score


def umap_spectral_domain(frequency_array, gene_score, n_dim=2):
    adata_gene = sc.AnnData(frequency_array).T
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.umap(adata_gene)
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    gene_score = gene_score.reindex(adata_gene.obs.index)
    adata_gene.obs = gene_score
    sc.pl.umap(adata_gene, color='spatially_variable')


def tsne_spectral_domain(frequency_array, gene_score, n_dims=2):
    adata_gene = sc.AnnData(frequency_array).T
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.tsne(adata_gene)
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    gene_score = gene_score.reindex(adata_gene.obs.index)
    adata_gene.obs = gene_score
    sc.pl.tsne(adata_gene, color='spatially_variable')


def fms_spectral_domain(frequency_array, gene_score, n_dims=2):
    adata_gene = sc.AnnData(frequency_array).T
    adata_gene.obsm['X_pca'] = frequency_array.transpose()
    gene_score = select_svg_normal(gene_score, num_sigma=3)
    gene_score = gene_score.reindex(adata_gene.obs.index)
    adata_gene.obs = gene_score
    sc.pl.pca(adata_gene, color='spatially_variable')


def pca_spatial_domain(adata, gene_score, n_dims=2):
    adata_gene = adata.copy()
    adata_gene = adata_gene.T
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    adata_gene.obs['spatially_variable'] = 0
    svg_index = (gene_score[gene_score['spatially_variable'] == 1]).index
    adata_gene.obs.loc[svg_index, 'spatially_variable'] = 1
    sc.pp.pca(adata_gene)
    sc.pl.pca(adata_gene, color='spatially_variable')


def umap_spatial_domain(adata, gene_score, n_dim=2):
    adata_gene = adata.copy()
    adata_gene = adata_gene.T
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    adata_gene.obs['spatially_variable'] = 0
    svg_index = (gene_score[gene_score['spatially_variable'] == 1]).index
    adata_gene.obs.loc[svg_index, 'spatially_variable'] = 1
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.umap(adata_gene)
    sc.pl.umap(adata_gene, color='spatially_variable')


def tsne_spatial_domain(adata, gene_score, n_dim=2):
    adata_gene = adata.copy()
    adata_gene = adata_gene.T
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    adata_gene.obs['spatially_variable'] = 0
    svg_index = (gene_score[gene_score['spatially_variable'] == 1]).index
    adata_gene.obs.loc[svg_index, 'spatially_variable'] = 1
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.tsne(adata_gene)
    sc.pl.tsne(adata_gene, color='spatially_variable')


def cal_mean_expression(adata, gene_list):
    tmp_adata = adata[:, gene_list].copy()
    if 'log1p' not in adata.uns_keys():
        tmp_adata = sc.pp.log1p(tmp_adata)
    mean_vector = tmp_adata.X.mean(axis=1)
    mean_vector = np.array(mean_vector).ravel()

    return mean_vector


def kneed_select_values(value_list, S=3, increasing=True):
    from kneed import KneeLocator
    x_list = list(range(1, 1 + len(value_list)))
    y_list = value_list.copy()
    if increasing:
        magic = KneeLocator(x=x_list,
                            y=y_list,
                            S=S)
    else:
        y_list = y_list[::-1].copy()
        magic = KneeLocator(x=x_list,
                            y=y_list,
                            direction='decreasing',
                            S=S,
                            curve='convex')
    return magic.elbow


def correct_pvalues_for_multiple_testing(pvalues,
                                         correction_type="Benjamini-Hochberg"):
    """
    Correct p-values to obtain the adjusted p-values

    Parameters
    ----------
    pvalues : list | 1-D array
        The original p values. It should be a list.
    correction_type : str, optional
        Method used to correct p values. The default is "Benjamini-Hochberg".

    Returns
    -------
    new_pvalues : array
        Corrected p values.

    """
    from numpy import array, empty
    pvalues = array(pvalues)
    n = int(pvalues.shape[0])
    new_pvalues = empty(n)
    if correction_type == "Bonferroni":
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            new_pvalues[i] = (n - rank) * pvalue
    elif correction_type == "Benjamini-Hochberg":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n / rank) * pvalue)
        for i in range(0, int(n) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]
    return new_pvalues


def permutation_signal(signal_array, num_permutation=1000):
    """
    Permutate gene signals in spatial domain randomly.

    Parameters
    ----------
    signal_array : list | array
        A one-dimensional array indicate gene expression on all spots.
    num_permutation : int, optional
        The number of permutation. The default is 1000.

    Returns
    -------
    total_signals : array
        The permuted gene expression signals.

    """
    signal_array = np.array(signal_array)
    total_signals = signal_array * np.ones((num_permutation,
                                            len(signal_array)))

    for i in range(num_permutation):
        total_signals[i, :] = np.random.permutation(total_signals[i, :])

    return total_signals


def significant_test_permutation(exp_mtx,
                                 gene_score,
                                 eigvals,
                                 eigvecs_T,
                                 num_permutaion=1000,
                                 num_pool=200,
                                 spec_norm='l1'):
    """
    To calculate p values for genes, permutate gene expression data and 
    calculate p values.

    Parameters
    ----------
    exp_mtx : 2D-array
        The count matrix of gene expressions. (spots * genes)
    gene_score : 1D-array
        The calculated gene scores. 
    eigvals : array
        The eigenvalues of Laplacian matrix.
    eigvecs_T : array
        The eigenvectors of Laplacian matrix.
    num_permutation : int, optional
        The number of permutations. The default is 1000.
    num_pool : int, optional
        The cores used for multiprocess calculation to accelerate speed. The default is 200.
    spec_norm : str, optional
        The method to normalize graph signals in spectral domain. The default  is 'l1'.

    Returns
    -------
    array
        The calculated p values.

    """
    from multiprocessing.dummy import Pool as ThreadPool
    from scipy.stats import mannwhitneyu

    def _test_by_permutaion(gene_index):
        (gene_index)
        graph_signal = exp_mtx[gene_index, :]
        total_signals = permutation_signal(signal_array=graph_signal, num_permutation=num_permutaion)
        frequency_array = np.matmul(eigvecs_T, total_signals.transpose())
        frequency_array = np.abs(frequency_array)
        if spec_norm != None:
            frequency_array = preprocessing.normalize(frequency_array,
                                                      norm=spec_norm,
                                                      axis=0)
        score_list = np.matmul(2 ** (-1 * eigvals), frequency_array)
        score_list = score_list / score_max
        pval = mannwhitneyu(score_list, gene_score[gene_index], alternative='less').pvalue
        return pval

    score_max = np.matmul(2 ** (-2 * eigvals), (1 / len(eigvals)) * \
                          np.ones(len(eigvals)))
    gene_index_list = list(range(exp_mtx.shape[0]))
    pool = ThreadPool(num_pool)
    res = pool.map(_test_by_permutaion, gene_index_list)

    return res


def test_significant_freq(freq_array,
                          cutoff,
                          num_pool=200):
    """
    Significance test by comparing the intensities in low frequency FMs and in high frequency FMs.

    Parameters
    ----------
    freq_array : array
        The graph signals of genes in frequency domain. 
    cutoff : int
        Watershed between low frequency signals and high frequency signals.
    num_pool : int, optional
        The cores used for multiprocess calculation to accelerate speed. The
        default is 200.

    Returns
    -------
    array
        The calculated p values.

    """
    from scipy.stats import ranksums
    from multiprocessing.dummy import Pool as ThreadPool

    def _test_by_feq(gene_index):
        freq_signal = freq_array[gene_index, :]
        freq_1 = freq_signal[:cutoff]
        freq_1 = freq_1[freq_1 > 0]
        freq_2 = freq_signal[cutoff:]
        freq_2 = freq_2[freq_2 > 0]
        if freq_1.size <= 80 or freq_2.size <= 80:
            freq_1 = np.concatenate((freq_1, freq_1, freq_1, freq_1))
            freq_2 = np.concatenate((freq_2, freq_2, freq_2, freq_2))
        if freq_1.size <= 120 or freq_2.size <= 120:
            freq_1 = np.concatenate((freq_1, freq_1, freq_1))
            freq_2 = np.concatenate((freq_2, freq_2, freq_2))
        if freq_1.size <= 160 or freq_2.size <= 160:
            freq_1 = np.concatenate((freq_1, freq_1))
            freq_2 = np.concatenate((freq_2, freq_2))
        pval = ranksums(freq_1, freq_2, alternative='greater').pvalue
        return pval

    gene_index_list = list(range(freq_array.shape[0]))
    pool = ThreadPool(num_pool)
    res = pool.map(_test_by_feq, gene_index_list)

    return res


def my_eigsh(args_tuple):
    """
    The function is used to multi-process calculate using Pool.

    Parameters
    ----------
    args_tuple : tupple
        The args_tupple contains three elements, that are, Lplacian matrix, k 
        and which.

    Returns
    -------
    (eigvals, eigvecs)

    """
    lap_mtx = args_tuple[0]
    k = args_tuple[1]
    which = args_tuple[2]
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx.astype(float),
                                       k=k,
                                       which=which)
    return ((eigvals, eigvecs))


def get_cos_similar(v1: list, v2: list):
    v3 = np.array(v1) + np.array(v2)
    return v3[v3 >= 2].size


def get_overlap_cs_core(cluster_collection):
    over_loop_cs_score = 0
    over_loop_cs_score_num = 0
    spot_shape = cluster_collection.shape[1]
    for index, _ in enumerate(cluster_collection):
        v1 = cluster_collection[index]
        if index + 1 <= cluster_collection.shape[0]:
            for value in cluster_collection[index + 1:]:
                over_loop_cs_score += get_cos_similar(value, v1)
                over_loop_cs_score_num += 1
    return (over_loop_cs_score / over_loop_cs_score_num / spot_shape) \
        if over_loop_cs_score_num != 0 else 1


def low_pass_enhancement(adata,
                         ratio_low_freq='infer',
                         ratio_neighbors='infer',
                         c=0.001,
                         spatial_info=['array_row', 'array_col'],
                         normalize_lap=False,
                         inplace=False):
    """
    Implement gene expressions with low-pass filter. After this step, the spatially variables genes will be smoother
    than the previous. The function can also be treated as denoising function. Note that the denoising results
    is related to spatial graph topology so that only the results of spatially variable genes are reasonable.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates of all spots should be found in
        adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with low frequencies. Indeed,
        the ratio_low_freq * sqrt(number of spots) low frequency FMs will be calculated. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when construct the KNN graph by spatial coordinates. Indeed, ratio_neighbors * sqrt(number of spots) / 2
        indicates the K. If 'infer', the parameter will be set to 1.0. The default is 'infer'.
    c: float, optional
        c balances the smoothness and difference with previous expression. A high can achieve better smoothness.
        c should be set to [0, 0.1].The default is 0.001.
    spatial_info : list | tuple | string, optional
        The column names of spatial coordinates in adata.obs_names or key in adata.obsm_keys() to obtain spatial
        information. The default is ['array_row', 'array_col'].
    normalize_lap : bool. optional
        Whether you need to normalize the Laplacian matrix. The default is False.
    inplace: bool, optional
        Whether you need to replace adata.X with the enhanced expression matrix. The default is False.
        

    Returns
    -------
    adata: anndata

    """
    import scipy.sparse as ss
    if ratio_low_freq == 'infer':
        if adata.shape[0] <= 800:
            num_low_frequency = min(20 * int(np.ceil(np.sqrt(adata.shape[0]))),
                                    adata.shape[0])
        elif adata.shape[0] <= 5000:
            num_low_frequency = 15 * int(np.ceil(np.sqrt(adata.shape[0])))
        elif adata.shape[0] <= 10000:
            num_low_frequency = 10 * int(np.ceil(np.sqrt(adata.shape[0])))
        else:
            num_low_frequency = 5 * int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * ratio_low_freq))

    if ratio_neighbors == 'infer':
        if adata.shape[0] <= 500:
            num_neighbors = 4
        else:
            num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 * ratio_neighbors))

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # Get Laplacian matrix according to coordinates 
    lap_mtx = get_laplacian_mtx(adata,
                                num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Fourier mode of low frequency
    num_low_frequency = min(num_low_frequency, adata.shape[0])
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx.astype(float),
                                       k=num_low_frequency,
                                       which='SM')

    # *********************** Graph Fourier Transform **************************
    # Calculate GFT
    eigvecs_t = eigvecs.transpose()
    if not ss.issparse(adata.X):
        exp_mtx = adata.X
    else:
        exp_mtx = adata.X.toarray()
    frequency_array = np.matmul(eigvecs_t, exp_mtx)
    # low-pass filter
    filter_list = [1 / (1 + c * eigv) for eigv in eigvals]
    # filter_list = [np.exp(-c * eigv) for eigv in eigvals]
    filter_array = np.matmul(np.diag(filter_list), frequency_array)
    filter_array = np.matmul(eigvecs, filter_array)
    filter_array[filter_array < 0] = 0

    # whether you need to replace original count matrix
    if inplace and not ss.issparse(adata.X):
        adata.X = filter_array
    elif inplace:
        import scipy.sparse as ss
        adata.X = ss.csr.csr_matrix(filter_array)

    return adata


def determine_frequency_ratio(adata,
                              low_end=5,
                              high_end=5,
                              ratio_neighbors='infer',
                              spatial_info=['array_row', 'array_col'],
                              normalize_lap=False):
    '''
    This function can choose the number of FMs automatically based on
    kneedle algorithm.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates of all spots should be found in
        adata.obs or adata.obsm.
    low_end : float, optional
        The range of low-frequency FMs. The default is 5. Note that the real cutoff is low_end * sqrt(n_spots).
    high_end : TYPE, optional
        The range of high-frequency FMs. The default is 5.Note that the real cutoff is low_end * sqrt(n_spots).
    ratio_neighbors : float, optional
        The ratio_neighbors will be used to determine the number of neighbors when construct the KNN graph by spatial
        coordinates. Indeed, ratio_neighbors * sqrt(number of spots) / 2 indicates the K. If 'infer', the parameter
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tuple | string, optional
        The column names of spatial coordinates in adata.obs_names or key
        in adata.obsm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    normalize_lap : bool, optional
        Whether you need to normalize the Laplacian matrix. The default is False.

    Returns
    -------
    low_cutoff : float
        The low_cutoff * sqrt(the number of spots) low-frequency FMs are 
        recommended in detecting svg.
    high_cutoff : float
        The high_cutoff * sqrt(the number of spots) low-frequency FMs are 
        recommended in detecting svg.

    '''
    # Determine the number of neighbors
    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4
    if adata.shape[0] > 15000 and low_end >= 3:
        low_end = 3
    if adata.shape[0] > 15000 and high_end >= 3:
        high_end = 3
    # Ensure gene index uniquely and all gene had expression  
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # *************** Construct graph and corresponding matrixs ***************
    lap_mtx = get_laplacian_mtx(adata,
                                num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)
    print("Obtain the Laplacian matrix")

    # Next, calculate the eigenvalues and eigenvectors of the Laplace matrix
    # Fourier bases of low frequency
    eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                           k=int(np.ceil(low_end * np.sqrt(adata.shape[0]))),
                                           which='SM')
    low_cutoff = np.ceil(kneed_select_values(eigvals_s) / np.sqrt(adata.shape[0]) * 1000) / 1000
    if low_cutoff >= low_end:
        low_cutoff = low_end
    if low_cutoff < 1:
        low_cutoff = 1
    if adata.shape[0] >= 40000 and low_cutoff <= 0.5:
        low_cutoff = 0.5
    num_low = int(np.ceil(np.sqrt(adata.shape[0]) * \
                          low_cutoff))
    eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                           k=int(np.ceil(high_end * np.sqrt(adata.shape[0]))),
                                           which='LM')
    high_cutoff = np.ceil(kneed_select_values(eigvals_l, increasing=False) / \
                          np.sqrt(adata.shape[0]) * 1000) / 1000
    if high_cutoff < 1:
        high_cutoff = 1
    if high_cutoff >= high_end:
        high_cutoff = high_end
    if adata.shape[0] >= 40000 and high_cutoff <= 0.5:
        high_cutoff = 0.5
    num_high = int(np.ceil(np.sqrt(adata.shape[0]) * \
                           high_cutoff))

    adata.uns['FMs_after_select'] = {'low_FMs_frequency': eigvals_s[:num_low],
                                     'low_FMs': eigvecs_s[:, :num_low],
                                     'high_FMs_frequency': eigvals_l[(len(eigvals_l) - num_high):],
                                     'high_FMs': eigvecs_l[:, (len(eigvals_l) - num_high):]}

    return low_cutoff, high_cutoff


def detect_svg(adata,
               ratio_low_freq='infer',
               ratio_high_freq='infer',
               ratio_neighbors='infer',
               spatial_info=['array_row', 'array_col'],
               normalize_lap=False,
               filter_peaks=True,
               S=6,
               cal_pval=True):
    """
    Rank genes according to GFT score to find spatially variable genes based on
    graph Fourier transform.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates could be found in adata.obs or
        adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequency FMs will be calculated. If 'infer', the ratio_low_freq will be
        set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs of
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots) 
        high frequency FMs will be calculated. If 'infer', the ratio_high_freq
        will be set to 1.0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when construct the KNN graph by spatial coordinates. Indeed, ratio_neighbors * sqrt(number of spots) / 2
         indicates the K. If 'infer', the parameter will be set to 1.0. The default is 'infer'.
    spatial_info : list | tuple | string, optional
        The column names of spatial coordinates in adata.obs_names or key
        in adata.varm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    normalize_lap : bool, optional
        Whether you need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domain, whether
        filter low peaks to stress the important peaks. The default is True.
    S: int, optional
        The sensitivity parameter in Kneedle algorithm. A large S will enable
        more genes identified as svgs according to gft_score. The default is 6.
    cal_pval : bool, optional
        Whether you need to calculate p val by mannwhitneyu. The default is False.
    Returns
    -------
    score_df : dataframe
        Return gene information.

    """
    # Ensure parameters
    if ratio_low_freq == 'infer':
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * ratio_high_freq))

    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4

    # Ensure gene index uniquely and all genes have expression  
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # Check dimensions
    if 'FMs_after_select' in adata.uns_keys():
        low_condition = (num_low_frequency == adata.uns['FMs_after_select']['low_FMs_frequency'].size)
        high_condition = (num_high_frequency == adata.uns['FMs_after_select']['high_FMs_frequency'].size)
    else:
        low_condition = False
        high_condition = False
    # ************ Construct graph and corresponding matrix *************
    lap_mtx = get_laplacian_mtx(adata, num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Next, calculate the eigenvalues and eigenvectors of the Laplacian
    # matrix as the Fourier modes with certain frequencies
    if not low_condition:
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
    else:
        eigvals_s, eigvecs_s = adata.uns['FMs_after_select']['low_FMs_frequency'], \
            adata.uns['FMs_after_select']['low_FMs']
        print('The precalculated low-frequency FMs are USED')
    if not high_condition:
        if num_high_frequency > 0:
            # Fourier bases of high frequency
            eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                                   k=num_high_frequency,
                                                   which='LM')
    else:
        eigvals_l, eigvecs_l = adata.uns['FMs_after_select']['high_FMs_frequency'], \
            adata.uns['FMs_after_select']['high_FMs']
        print('The precalculated high-frequency FMs are USED')
    if num_high_frequency > 0:
        # eigenvalues
        eigvals = np.concatenate((eigvals_s, eigvals_l))
        # eigenvectors
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)
    else:
        eigvals = eigvals_s
        eigvecs = eigvecs_s

    # ************************ Graph Fourier Transform *************************
    # Calculate GFT
    eigvecs_t = eigvecs.transpose()
    if type(adata.X) == np.ndarray:
        exp_mtx = preprocessing.scale(adata.X)
    else:
        exp_mtx = preprocessing.scale(adata.X.toarray())

    frequency_array = np.matmul(eigvecs_t, exp_mtx)
    frequency_array = np.abs(frequency_array)

    # Filter noise peaks
    if filter_peaks:
        frequency_array_thres_low = np.quantile(frequency_array[:num_low_frequency, :], q=0.5, axis=0)
        frequency_array_thres_high = np.quantile(frequency_array[num_low_frequency:, :], q=0.5, axis=0)
        for j in range(frequency_array.shape[1]):
            frequency_array[:num_low_frequency, :][frequency_array[:num_low_frequency, j] <= \
                                                   frequency_array_thres_low[j], j] = 0
            frequency_array[num_low_frequency:, :][frequency_array[num_low_frequency:, j] <= \
                                                   frequency_array_thres_high[j], j] = 0

    frequency_array = preprocessing.normalize(frequency_array,
                                              norm='l1',
                                              axis=0)

    eigvals = np.abs(eigvals)
    eigvals_weight = np.exp(-1 * eigvals)
    score_list = np.matmul(eigvals_weight, frequency_array)
    score_ave = np.matmul(eigvals_weight, (1 / len(eigvals)) * \
                          np.ones(len(eigvals)))
    score_list = score_list / score_ave
    print("Graph Fourier Transform finished!")

    # Rank genes according to smooth score
    adata.var["gft_score"] = score_list
    score_df = adata.var["gft_score"]
    score_df = pd.DataFrame(score_df)
    score_df = score_df.sort_values(by="gft_score", ascending=False)
    score_df.loc[:, "svg_rank"] = range(1, score_df.shape[0] + 1)
    adata.var["svg_rank"] = score_df.reindex(adata.var_names).loc[:, "svg_rank"]
    print("svg ranking could be found in adata.var['svg_rank']")

    # Determine cutoff of gft_score
    from kneed import KneeLocator
    magic = KneeLocator(score_df.svg_rank.values,
                        score_df.gft_score.values,
                        direction='decreasing',
                        curve='convex',
                        S=S)
    score_df['cutoff_gft_score'] = False
    score_df['cutoff_gft_score'][:(magic.elbow + 1)] = True
    adata.var['cutoff_gft_score'] = score_df['cutoff_gft_score']
    print("""The spatially variable genes judged by gft_score could be found 
          in adata.var['cutoff_gft_score']""")
    adata.varm['freq_domain_svg'] = frequency_array.transpose()
    print("""Gene signals in frequency domain when detect svgs could be found
          in adata.varm['freq_domain_svg']""")
    adata.uns["identify_svg_data"] = {}
    adata.uns["identify_svg_data"]['frequencies_low'] = eigvals_s
    adata.uns["identify_svg_data"]['frequencies_high'] = eigvals_l
    adata.uns["identify_svg_data"]['fms_low'] = eigvecs_s
    adata.uns["identify_svg_data"]['fms_high'] = eigvecs_l

    if cal_pval:
        if num_high_frequency == 0:
            raise ValueError("ratio_high_freq should be greater than 0")
        pval_list = test_significant_freq(
            freq_array=adata.varm['freq_domain_svg'],
            cutoff=num_low_frequency)
        from statsmodels.stats.multitest import multipletests
        qval_list = multipletests(np.array(pval_list), method='fdr_by')[1]
        adata.var['pvalue'] = pval_list
        adata.var['fdr'] = qval_list
        score_df = adata.var.loc[score_df.index, :].copy()

    return score_df


def calculate_frequency_domain(adata,
                               ratio_low_freq='infer',
                               ratio_high_freq='infer',
                               ratio_neighbors='infer',
                               spatial_info=['array_row', 'array_col'],
                               return_freq_domain=True,
                               normalize_lap=False,
                               filter_peaks=False):
    """
    Obtain gene signals in frequency/spectral domain for all genes in 
    adata.var_names.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates could be found in adata.obs
        or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with low frequencies. Indeed,
        the ratio_low_freq * sqrt(number of spots) low frequency FMs will be calculated. If 'infer',
        the ratio_low_freq will be set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with high frequencies. Indeed,
        the ratio_high_freq * sqrt(number of spots) high frequency FMs will be calculated. If 'infer',
        the ratio_high_freq will be set to 0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors when construct the KNN graph by spatial
        coordinates. Indeed, ratio_neighbors * sqrt(number of spots) / 2 indicates the K. If 'infer', the parameter
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tuple | str, optional
        The column names of spatial coordinates in adata.obs_keys() or a key in adata.obsm_keys.
        The default is ['array_row','array_col'].
    return_freq_domain : bool, optional
        Whether you need to return gene signals in frequency domain. The default is True.
    normalize_lap : bool, optional
        Whether you need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domain, whether filter low peaks to stress the important
        peaks. The default is False.

    Returns
    -------
    If return_freq_domain, return DataFrame, the index indicates the gene and 
    the columns indicates corresponding frequencies/smoothness.

    """
    # Critical parameters
    # Ensure parameters
    if ratio_low_freq == 'infer':
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * ratio_high_freq))
    if adata.shape[0] >= 10000:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))

    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4

    # Ensure gene index uniquely and all gene had expression
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # ************** Construct graph and corresponding matrix ****************
    lap_mtx = get_laplacian_mtx(adata,
                                num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Calculate the eigenvalues and eigenvectors of the Laplace matrix
    np.random.seed(123)
    if num_high_frequency > 0:
        # Fourier modes of low frequency
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
        # Fourier mode of high frequency
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        eigvals = np.concatenate((eigvals_s, eigvals_l))  # eigenvalues
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)  # eigenvectors
    else:
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
        eigvecs = eigvecs_s
        eigvals = eigvals_s

    # ************************Graph Fourier Transform***************************
    # Calculate GFT
    eigvecs = eigvecs.transpose()
    if not ss.issparse(adata.X):
        exp_mtx = adata.X.copy()
    else:
        exp_mtx = adata.X.toarray().copy()
    exp_mtx = preprocessing.scale(exp_mtx, axis=0)
    frequency_array = np.matmul(eigvecs, exp_mtx)
    # Filter noise peaks
    if filter_peaks:
        frequency_array_thres = np.median(frequency_array, axis=0)
        for j in range(adata.shape[1]):
            frequency_array[frequency_array[:, j] <= \
                            frequency_array_thres[j], j] = 0
    # Spectral domain normalization
    frequency_array = preprocessing.normalize(frequency_array,
                                              norm='l1', axis=0)

    # ********************** Results of GFT ***********************************
    frequency_df = pd.DataFrame(frequency_array, columns=adata.var_names,
                                index=['low_spec_' + str(low) \
                                       for low in range(1, num_low_frequency + 1)] \
                                      + ['high_spec_' + str(high) \
                                         for high in range(1, num_high_frequency + 1)])
    adata.varm['freq_domain'] = frequency_df.transpose()
    adata.uns['frequencies'] = eigvals

    if return_freq_domain:
        return frequency_df


def freq2umap(adata,
              ratio_low_freq='infer',
              ratio_high_freq='infer',
              ratio_neighbors='infer',
              spatial_info=['array_row', 'array_col'],
              normalize_lap=False,
              filter_peaks=False):
    """
    Obtain gene signals in frequency/spectral domain for all genes in 
    adata.var_names and reduce dimension to 2 by UMAP.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates could be found in adata.obs
        or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with low frequencies. Indeed,
        the ratio_low_freq * sqrt(number of spots) low frequency FMs will be calculated. If 'infer',
        the ratio_low_freq will be set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with high frequencies. Indeed,
        the ratio_high_freq * sqrt(number of spots) high frequency FMs will be calculated. If 'infer',
        the ratio_high_freq will be set to 0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors when construct the KNN graph by spatial
        coordinates. Indeed, ratio_neighbors * sqrt(number of spots) / 2 indicates the K. If 'infer', the parameter
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tuple | str, optional
        The column names of spatial coordinates in adata.obs_keys() or in adata.obsm_keys.
        The default is ['array_row','array_col'].
    normalize_lap : bool, optional
        Whether you need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domain, whether filter low peaks to stress the
        important peaks. The default is False.

    """
    if 'svg_rank' not in adata.var.columns:
        assert KeyError("adata.var['svg_rank'] is not available. Please run SpaGFT.rank_gene_smooth(adata) firstly.")
    ftup_adata = adata.copy()
    if 'log1p' in adata.uns_keys():
        ftup_adata.uns.pop('log1p')
    ftup_adata.X = adata.raw[:, adata.var_names].X
    sc.pp.log1p(ftup_adata)
    calculate_frequency_domain(ftup_adata,
                               ratio_low_freq=ratio_low_freq,
                               ratio_high_freq=ratio_high_freq,
                               ratio_neighbors=ratio_neighbors,
                               spatial_info=spatial_info,
                               return_freq_domain=False,
                               normalize_lap=normalize_lap,
                               filter_peaks=filter_peaks)
    adata.varm['gft_umap_svg'] = ftup_adata.varm['gft_umap']


def identify_ftu(adata,
                 svg_list='infer',
                 ratio_fms='infer',
                 ratio_neighbors=2,
                 spatial_info=['array_row', 'array_col'],
                 n_neighbors=15,
                 resolution=1,
                 weight_by_freq=False,
                 normalize_lap=False,
                 random_state=0,
                 **kwargs):
    """
    After identifying spatially variable genes, this function will group these spatially variable genes sharing common
    spatial patterns.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinates could be found in adata.obs
        or adata.obsm.
    svg_list : list, optional
        The genes in svg_list will be grouped based on spatial patterns. The default is 'infer'.
    ratio_fms : float, optional
        The ratio_fms will be used to determine the number of the FMs with low frequencies. Indeed,
        the ratio_fms * sqrt(number of spots) low frequency FMs will be calculated. If 'infer',
        the ratio_low_freq will be determined automatically. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors when construct the KNN graph by spatial
        coordinates. Indeed, ratio_neighbors * sqrt(number of spots) / 2 indicates the K. If 'infer', the parameter
        will be set to 2.0. The default is 2.
    spatial_info : list | tuple | str, optional
        The column names of spatial coordinates in adata.obs_keys() or a key in adata.obsm_keys.
         The default is ['array_row','array_col'].
    n_neighbors : int, optional
        The neighbors in gene similarity graph to perform louvain algorithm. 
        The default is 15.
    resolution : float | list | tuple, optional
        The resolution parameter in louvain algorithm. If resolution is float, resolution will be used directly.
        If resolution is a list, each value in this list will be used and the best value will be determined
        automatically. If resolution is tuple, it should be (start, end, step), and it is similar to a list.
        The default is 1.
    weight_by_freq : bool, optional
        Whether you need to weight FC according to frequencies. The default is False.
    normalize_lap : bool, optional
        Whether you need to normalize laplacian matrix. The default is false.
    random_state : int, optional
        The randomstate. The default is 0.
    **kwargs : kwargs 
        The parameters used in louvain algorithms and user can seek help in  sc.tl.louvain.

    Returns
    -------
    DataFrame
        The ftu information after identification.

    """
    # Find ftu by grouping Spatially variable genes with similar 
    # spatial patterns according to louvain algorithm.
    # Check conditions and determine parameters
    if isinstance(resolution, float) or isinstance(resolution, int):
        single_resolution = True
    else:
        single_resolution = False
    if isinstance(resolution, tuple) and len(resolution) == 3:
        start, end, step = resolution
        resolution = np.arange(start, end, step).tolist()
    elif isinstance(resolution, tuple):
        raise ValueError("""when resolution is a tuple, it should be (start, end, step)""")
    if isinstance(resolution, np.ndarray):
        resolution = resolution.tolist()

    assert isinstance(resolution, float) or isinstance(resolution, int) \
           or isinstance(resolution, list), 'please input resolution with type of float, int, list or tuple'

    if 'svg_rank' not in adata.var.columns:
        assert KeyError("adata.var['svg_rank'] is not available. Please run SpaGFT.detect_svg(adata) firstly.")
    if ratio_fms == 'infer':
        if adata.shape[0] <= 500:
            ratio_fms = 4
        elif adata.shape[0] <= 10000:
            ratio_fms = 2
        else:
            ratio_fms = 1
    if isinstance(svg_list, str):
        if svg_list == 'infer':
            gene_score = adata.var.sort_values(by='svg_rank')
            adata = adata[:, gene_score.index]
            svg_list = adata.var[adata.var.cutoff_gft_score][adata.var.fdr < 0.05].index.tolist()
    ftup_adata = adata[:, svg_list].copy()
    if 'log1p' in adata.uns_keys():
        ftup_adata.uns.pop('log1p')
    ftup_adata.X = adata[:, svg_list].X.copy()
    calculate_frequency_domain(ftup_adata,
                               ratio_low_freq=ratio_fms,
                               ratio_high_freq=0,
                               ratio_neighbors=ratio_neighbors,
                               spatial_info=spatial_info,
                               return_freq_domain=False,
                               normalize_lap=normalize_lap,
                               filter_peaks=False)
    # Create new anndata to store freq domain information
    gft_adata = sc.AnnData(ftup_adata.varm['freq_domain'])
    if weight_by_freq:
        weight_list = 1 / (1 + 0.01 * ftup_adata.uns['frequencies'])
        gft_adata.X = np.multiply(gft_adata.X, weight_list)
        gft_adata.X = preprocessing.normalize(gft_adata.X, norm='l1')
    # clustering
    gft_adata = gft_adata[svg_list, :]
    adata.uns['detect_ftu_data'] = {}
    adata.uns['detect_ftu_data']['freq_domain_svgs'] = ftup_adata[:, svg_list].varm['freq_domain']
    if gft_adata.shape[1] >= 400:
        sc.pp.pca(adata)
        sc.pp.neighbors(gft_adata, n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(gft_adata, n_neighbors=n_neighbors, use_rep='X')

    # Determining the resolution data type, if resolution is the type of list, 
    # we will select the optimal resolution
    if isinstance(resolution, list):
        # The minimum value of cosine similarity of overlap: 
        # count_ftus_select_cs_score
        count_ftus_select_cs_score = 1
        best_resolution = None
        overlap_scores = {}
        # Iterate through the list of resolution and calculate the 
        # clustering results
        for resolution_index, resolution_value in enumerate(resolution):
            gft_adata_current = gft_adata.copy()
            sc.tl.louvain(gft_adata_current,
                          resolution=resolution_value,
                          random_state=random_state,
                          key_added='louvain',
                          **kwargs)

            gft_adata_current.obs.louvain = [str(eval(i_ftu) + 1) for i_ftu in gft_adata_current.obs.louvain.tolist()]
            gft_adata_current.obs.louvain = pd.Categorical(gft_adata_current.obs.louvain)

            # ftu pseudo expression
            all_ftus_current = gft_adata_current.obs.louvain.cat.categories
            ftu_df_current = pd.DataFrame(0,
                                          index=ftup_adata.obs_names,
                                          columns='ftu_' + all_ftus_current)
            # Calculate the clustering of each ftu
            for ftu in all_ftus_current:
                pseudo_exp = ftup_adata[:,
                             gft_adata_current.obs.louvain[gft_adata_current.obs.louvain == ftu].index].X.sum(axis=1)
                pseudo_exp = np.ravel(pseudo_exp)
                # Calculate the clustering results
                predict_ftu = KMeans(n_clusters=2, random_state=random_state).fit_predict(pseudo_exp.reshape(-1, 1))

                # Correct clustering results
                pseudo_exp_median = np.median(pseudo_exp)
                pseudo_exp_cluster = np.where(pseudo_exp > pseudo_exp_median, 1, 0)

                cluster_middle_param = sum(abs(predict_ftu - pseudo_exp_cluster))
                cluster_middle_param_reverse = sum(abs(predict_ftu - abs(pseudo_exp_cluster - 1)))
                if cluster_middle_param > cluster_middle_param_reverse:
                    predict_ftu = abs(predict_ftu - 1)
                ftu_df_current['ftu_' + str(ftu)] = predict_ftu

            # Correct cosine similarity of overlap for clustering results
            overlap_cs_score = get_overlap_cs_core(ftu_df_current.values.T)
            print("""resolution: %.3f;  """ % resolution_value + """score: %.4f""" % overlap_cs_score)
            overlap_scores['res_' + '%.3f' % resolution_value] = np.round(overlap_cs_score * 1e5) / 1e5
            # select the optimal resolution
            if count_ftus_select_cs_score > overlap_cs_score or resolution_index == 0:
                count_ftus_select_cs_score = overlap_cs_score
                best_resolution = resolution_value
            resolution = best_resolution

    # Next, clustering genes for given resolution
    sc.tl.louvain(gft_adata, resolution=resolution, random_state=random_state,
                  **kwargs, key_added='louvain')
    sc.tl.umap(gft_adata)
    adata.uns['detect_ftu_data']['gft_umap_ftu'] = pd.DataFrame(gft_adata.obsm['X_umap'],
                                                                index=gft_adata.obs.index,
                                                                columns=['UMAP_1', 'UMAP_2'])
    gft_adata.uns['gft_genes_ftu'] = [str(eval(i_ftu) + 1) for i_ftu in gft_adata.obs.louvain.tolist()]
    gft_adata.obs.louvain = [str(eval(i_ftu) + 1) for i_ftu in gft_adata.obs.louvain.tolist()]
    gft_adata.obs.louvain = pd.Categorical(gft_adata.obs.louvain)
    adata.var['ftu'] = 'None'
    adata.var.loc[gft_adata.obs_names, 'ftu'] = gft_adata.obs.louvain
    adata.var['ftu'] = pd.Categorical(adata.var['ftu'])

    # ftu pseudo expression
    all_ftus = gft_adata.obs.louvain.cat.categories
    ftu_df = pd.DataFrame(0, index=adata.obs_names, columns='ftu_' + all_ftus)
    pseudo_df = pd.DataFrame(0, index=adata.obs_names, columns='ftu_' + all_ftus)
    for ftu in all_ftus:
        pseudo_exp = ftup_adata[:,
                     gft_adata.obs.louvain[gft_adata.obs.louvain == ftu].index].X.sum(axis=1)
        pseudo_exp = np.ravel(pseudo_exp)
        pseudo_df['ftu_' + str(ftu)] = pseudo_exp.copy()
        predict_ftu = KMeans(n_clusters=2, random_state=random_state).fit_predict(pseudo_exp.reshape(-1, 1))

        pseudo_exp_median = np.median(pseudo_exp)
        pseudo_exp_cluster = np.where(pseudo_exp > pseudo_exp_median, 1, 0)

        cluster_middle_param = sum(abs(predict_ftu - pseudo_exp_cluster))
        cluster_middle_param_reverse = sum(abs(predict_ftu - abs(pseudo_exp_cluster - 1)))
        if cluster_middle_param > cluster_middle_param_reverse:
            predict_ftu = abs(predict_ftu - 1)
        ftu_df['ftu_' + str(ftu)] = predict_ftu
        adata.obsm['ftu_pseudo_expression'] = pseudo_df.copy()

    ftu_df = ftu_df.astype(str)
    adata.obsm['ftu_binary'] = ftu_df.copy()
    # Obtain freq signal
    freq_signal_ftu_df = pd.DataFrame(0, index=ftu_df.columns,
                                      columns=ftup_adata.varm['freq_domain'].columns)

    for ftu in all_ftus:
        ftu_gene_list = gft_adata.obs.louvain[gft_adata.obs.louvain == ftu].index
        freq_signal = ftup_adata.varm['freq_domain'].loc[ftu_gene_list,
                      :].sum(axis=0)
        freq_signal = np.abs(freq_signal)
        freq_signal = freq_signal / sum(freq_signal)
        freq_signal_ftu_df.loc['ftu_' + ftu, :] = freq_signal
    adata.uns['detect_ftu_data']['freq_signal_ftu'] = freq_signal_ftu_df
    adata.uns['detect_ftu_data']['low_freq_domain_svg'] = pd.DataFrame(gft_adata.X.copy(),
                                                                       index=gft_adata.obs_names,
                                                                       columns=['low_freq_' + str(i + 1) \
                                                                                for i in range(gft_adata.shape[1])])
    if not single_resolution:
        adata.uns['detect_ftu_data']['overlap_curve'] = pd.DataFrame(overlap_scores, index=['score'])
        adata.uns['detect_ftu_data']['overlap_curve'] = adata.uns['detect_ftu_data']['overlap_curve'].transpose()

    return adata.var.loc[svg_list, :].copy(), adata


def gene_freq_signal(adata,
                     gene,
                     domain='freq_domain_svg',
                     figsize=(6, 2),
                     dpi=100,
                     colors=['#CA1C1C', '#345591'],
                     save_path=None,
                     return_fig=False,
                     **kwargs):
    if isinstance(gene, str):
        freq_signal = adata[:, gene].varm[domain]
        freq_signal = np.ravel(freq_signal)
        plt.figure(figsize=figsize, dpi=dpi)
        low = list(range(adata.uns['identify_svg_data']['fms_low'].shape[1]))
        plt.bar(low, freq_signal[low], color=colors[0])
        high = list(range(len(low), freq_signal.size))
        plt.bar(high, freq_signal[high], color=colors[1])
        ax = plt.gca()
        ax.set_ylabel("signal")
        ax.spines['right'].set_color("none")
        ax.spines['top'].set_color("none")
        y_max = max(freq_signal)
        plt.ylim(0, y_max * 1.1)
        plt.xlim(0, freq_signal.size)
        plt.title("Gene: " + gene)
        plt.show()
        if save_path is not None:
            plt.savefig(f"{save_path}")
        if return_fig:
            return ax
    elif isinstance(gene, list) or isinstance(gene, np.ndarray):
        row = len(gene)
        fig = plt.figure(dpi=350,
                         constrained_layout=True,
                         figsize=(8, row * 2)
                         )

        gs = GridSpec(row, 1,
                      figure=fig)
        ax_list = []
        for index, value in enumerate(gene):
            ax = fig.add_subplot(gs[index, 0])
            freq_signal = adata[:, value].varm[domain]
            freq_signal = np.ravel(freq_signal)
            low = list(range(adata.uns['identify_svg_data']['fms_low'].shape[1]))
            plt.bar(low, freq_signal[low], color=colors[0])
            high = list(range(len(low), freq_signal.size))
            plt.bar(high, freq_signal[high], color=colors[1])
            ax.set_ylabel("signal")
            ax.spines['right'].set_color("none")
            ax.spines['top'].set_color("none")
            y_max = max(freq_signal)
            plt.ylim(0, y_max * 1.1)
            plt.xlim(0, freq_signal.size)
            plt.title("Gene: " + value)
            ax_list.append(ax)

        if save_path is not None:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list


def ftu_freq_signal(adata,
                    ftu="ftu_1",
                    domain='freq_domain_svg',
                    figsize=(8, 2),
                    dpi=100,
                    color='#CA1C1C',
                    y_range=None,
                    xy_axis=True,
                    return_fig=False,
                    save_path=None,
                    **kwargs):
    if isinstance(ftu, str):
        # fig, ax = plt.subplots()
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        freq_signal = \
            adata.uns['detect_ftu_data']['freq_signal_ftu'].loc[ftu, :].values

        plt.bar(range(freq_signal.size), freq_signal, color=color)
        plt.grid(False)

        plt.title(ftu)
        ax.set_ylabel("signal")
        ax.spines['right'].set_color("none")
        ax.spines['top'].set_color("none")
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        else:
            plt.ylim(0, max(freq_signal) * 1.1)
        plt.xlim(0, freq_signal.size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("signal")
        if not xy_axis:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if save_path is not None:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax

    elif isinstance(ftu, list) or isinstance(ftu, np.ndarray):
        row = len(ftu)
        fig = plt.figure(dpi=350,
                         constrained_layout=True,
                         figsize=(8, row * 2)
                         )

        gs = GridSpec(row, 1,
                      figure=fig)
        ax_list = []
        for index, value in enumerate(ftu):
            ax = fig.add_subplot(gs[index, 0])
            freq_signal = \
                adata.uns['detect_ftu_data']['freq_signal_ftu'].loc[value,
                :].values

            plt.bar(range(freq_signal.size), freq_signal, color=color)
            plt.grid(False)

            plt.title(value)
            ax.set_ylabel("signal")
            ax.spines['right'].set_color("none")
            ax.spines['top'].set_color("none")
            if y_range != None:
                plt.ylim(y_range[0], y_range[1])
            else:
                plt.ylim(0, max(freq_signal) * 1.1)
            plt.xlim(0, freq_signal.size)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("signal")
            if not xy_axis:
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            ax_list.append(ax)

        if save_path is not None:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list


def _subftu_freq_signal(adata,
                        subftu,
                        figsize=(6, 2),
                        dpi=100,
                        color='#CA1C1C',
                        y_range=[0, 0.1],
                        return_fig=False, **kwargs):
    # Show the frequency signal
    freq_signal = adata.uns['freq_signal_subftu'].loc[subftu, :].values
    plt.figure(figsize=figsize, dpi=dpi)
    low = list(range(len(freq_signal)))
    plt.bar(low, freq_signal, color=color)
    ax = plt.gca()
    plt.grid(False)
    ax.set_ylabel("signal")
    ax.spines['right'].set_color("none")
    ax.spines['top'].set_color("none")
    plt.ylim(y_range[0], y_range[1])
    plt.xlim(0, freq_signal.size)
    plt.title(subftu)
    plt.show()
    if return_fig:
        return ax


def gene_signal_umap(adata,
                     svg_list,
                     colors=['#C71C1C', '#BEBEBE'],
                     n_neighbors=15,
                     random_state=0,
                     return_fig=False,
                     save_path=None,
                     **kwargs):
    """
    The UMAP of svgs and non-svgs.

    Parameters
    ----------
    adata : anndata
        The spatial dataset.
    svg_list : list
        The svg list.
    colors : list, optional
        The colors corresponding to svgs and non-svgs. 
        The default is ['#C71C1C', '#BEBEBE'].
    n_neighbors : int, optional
        The neighbors when construct gene graph for umap.
        The default is 15.
    random_state : int, optional
        The ramdom state. The default is 0.
    return_fig : bool, optional
        Whether you need to return figure. The default is False.
    save_path : system path | None, optional
        The path including filename to save the figure.

    Returns
    -------
    fig : matploblib axes
        The figure.

    """
    low_length = adata.uns['identify_svg_data']['frequencies_low'].shape[0]
    freq_domain = adata.varm['freq_domain_svg'][:, :low_length].copy()
    freq_domain = preprocessing.normalize(freq_domain, norm='l1')
    freq_domain = pd.DataFrame(freq_domain)
    freq_domain.index = adata.var_names
    umap_adata = sc.AnnData(freq_domain)
    sc.pp.neighbors(umap_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.umap(umap_adata, random_state=0)
    adata.varm['freq_umap_svg'] = umap_adata.obsm['X_umap']
    print("""The umap coordinates of genes when identify svgs could be found in 
          adata.varm['freq_umap_svg']""")
    # svg_list
    umap_adata.obs['SpaGFT'] = 'Non-svgs'
    umap_adata.obs.loc[svg_list, 'SpaGFT'] = 'svgs'
    umap_adata.obs['SpaGFT'] = pd.Categorical(umap_adata.obs['SpaGFT'],
                                              categories=['svgs', 'Non-svgs'],
                                              ordered=True)
    umap_adata.uns['SpaGFT_colors'] = colors
    fig = sc.pl.umap(umap_adata, color="SpaGFT", return_fig=return_fig,
                     **kwargs)

    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()
    if return_fig:
        return fig


def _scatter_gene_distri(adata,
                         gene,
                         size=3,
                         shape='h',
                         cmap='magma',
                         spatial_info=['array_row', 'array_col'],
                         coord_ratio=0.7,
                         return_plot=False):
    if gene in adata.obs.columns:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata.obs.loc[:, gene].values,
                                   index=adata.obs_names,
                                   columns=[gene])
        else:
            plot_df = pd.DataFrame(adata.obs.loc[:, gene],
                                   index=adata.obs_names,
                                   columns=gene)
        if spatial_info in adata.obsm_keys():
            plot_df['x'] = adata.obsm[spatial_info][:, 0]
            plot_df['y'] = adata.obsm[spatial_info][:, 1]
        elif set(spatial_info) <= set(adata.obs.columns):
            plot_coor = adata.obs
            plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
            plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values

        if isinstance(gene, str):
            base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y',
                                                            fill=gene),
                                               shape=shape, stroke=0.1,
                                               size=size) +
                         xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                         ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                         scale_fill_cmap(cmap_name=cmap) +
                         coord_equal(ratio=coord_ratio) +
                         theme_classic() +
                         theme(legend_position=('right'),
                               legend_background=element_blank(),
                               legend_key_width=4,
                               legend_key_height=50)
                         )
            print(base_plot)
        else:
            for i in gene:
                base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y',
                                                                fill=gene),
                                                   shape=shape, stroke=0.1,
                                                   size=size) +
                             xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                             ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                             scale_fill_cmap(cmap_name=cmap) +
                             coord_equal(ratio=coord_ratio) +
                             theme_classic() +
                             theme(legend_position=('right'),
                                   legend_background=element_blank(),
                                   legend_key_width=4,
                                   legend_key_height=50)
                             )
                print(base_plot)

        return
    if ss.issparse(adata.X):
        plot_df = pd.DataFrame(adata.X.todense(), index=adata.obs_names,
                               columns=adata.var_names)
    else:
        plot_df = pd.DataFrame(adata.X, index=adata.obs_names,
                               columns=adata.var_names)
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 0]
        plot_df['y'] = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
    plot_df['radius'] = size
    plot_df = plot_df.sort_values(by=gene, ascending=True)
    if isinstance(gene, str):
        base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y',
                                                        fill=gene),
                                           shape=shape, stroke=0.1,
                                           size=size) +
                     xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                     ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                     scale_fill_cmap(cmap_name=cmap) +
                     coord_equal(ratio=coord_ratio) +
                     theme_classic() +
                     theme(legend_position=('right'),
                           legend_background=element_blank(),
                           legend_key_width=4,
                           legend_key_height=50)
                     )
        print(base_plot)
    else:
        for i in gene:
            base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y',
                                                            fill=gene),
                                               shape=shape, stroke=0.1,
                                               size=size) +
                         xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                         ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                         scale_fill_cmap(cmap_name=cmap) +
                         coord_equal(ratio=coord_ratio) +
                         theme_classic() +
                         theme(legend_position=('right'),
                               legend_background=element_blank(),
                               legend_key_width=4,
                               legend_key_height=50)
                         )
            print(base_plot)
    if return_plot:
        return base_plot


def _umap_svg_cluster(adata,
                      svg_list=None,
                      size=None,
                      coord_ratio=1,
                      return_plot=True):
    if svg_list == None:
        ftup_df = adata.var.copy()
        svg_list = ftup_df[ftup_df.cutoff_gft_score][ftup_df.fdr < 0.05].index
    plot_df = adata.uns['gft_umap_ftu']
    plot_df = pd.DataFrame(plot_df)
    plot_df.index = adata.var_names
    plot_df.columns = ['UMAP_1', 'UMAP_2']

    plot_df.loc[svg_list, 'gene'] = 'svg'
    plot_df['gene'] = pd.Categorical(plot_df['gene'],
                                     categories=['svg', 'Non-svg'],
                                     ordered=True)
    plot_df['radius'] = size
    # plot
    if size is None:
        base_plot = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2', fill='gene')) +
                     geom_point(color='white', stroke=0.25) +
                     scale_fill_manual(values=colors) +
                     theme_classic() +
                     coord_equal(ratio=coord_ratio))
    else:
        base_plot = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2', fill='gene')) +
                     geom_point(size=size, color='white', stroke=0.25) +
                     scale_fill_manual(values=colors) +
                     theme_classic() +
                     coord_equal(ratio=coord_ratio))
    print(base_plot)
    if return_plot:
        return base_plot


def _scatter_ftu_binary(adata,
                        ftu,
                        size=3,
                        shape='h',
                        spatial_info=['array_row', 'array_col'],
                        colors=['#CA1C1C', '#CCCCCC'],
                        coord_ratio=0.7,
                        return_plot=False):
    if '-' in ftu:
        ftu = 'ftu-' + ftu.split('-')[0] + "_subftu-" + ftu.split('-')[1]
        plot_df = adata.obsm['subftu_binary']
    else:
        ftu = 'ftu_' + ftu
        plot_df = adata.obsm['ftu_binary']
    plot_df = pd.DataFrame(plot_df)
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 0]
        plot_df['y'] = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[ftu]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
    plot_df['radius'] = size
    plot_df[ftu] = plot_df[ftu].values.astype(int)
    plot_df[ftu] = plot_df[ftu].values.astype(str)
    plot_df[ftu] = pd.Categorical(plot_df[ftu],
                                  categories=['1', '0'],
                                  ordered=True)
    base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y', fill=ftu),
                                       shape=shape, stroke=0.1, size=size) +
                 xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                 ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                 scale_fill_manual(values=colors) +
                 coord_equal(ratio=coord_ratio) +
                 theme_classic() +
                 theme(legend_position=('right'),
                       legend_background=element_blank(),
                       legend_key_width=4,
                       legend_key_height=50)
                 )
    print(base_plot)
    if return_plot:
        return base_plot


def umap_svg(adata,
             svg_list=None,
             colors=['#CA1C1C', '#CCCCCC'],
             size=2,
             coord_ratio=0.7,
             return_plot=False):
    if 'gft_umap_svg' not in adata.varm_keys():
        raise KeyError("Please run SpaGFT.calculate_frequency_domain firstly")
    plot_df = adata.varm['gft_umap_svg']
    plot_df = pd.DataFrame(plot_df)
    plot_df.index = adata.var_names
    plot_df.columns = ['UMAP_1', 'UMAP_2']
    plot_df['gene'] = 'Non-svg'
    if svg_list == None:
        ftup_df = adata.var.copy()
        svg_list = ftup_df[ftup_df.cutoff_gft_score][ftup_df.fdr < 0.05].index
    plot_df.loc[svg_list, 'gene'] = 'svg'
    plot_df['gene'] = pd.Categorical(plot_df['gene'],
                                     categories=['svg', 'Non-svg'],
                                     ordered=True)
    plot_df['radius'] = size
    # plot
    base_plot = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2', fill='gene')) +
                 geom_point(size=size, color='white', stroke=0.25) +
                 scale_fill_manual(values=colors) +
                 theme_classic() +
                 coord_equal(ratio=coord_ratio))
    print(base_plot)
    if return_plot:
        return base_plot


def _visualize_fms(adata,
                   rank=1,
                   low=True,
                   size=3,
                   cmap='magma',
                   spatial_info=['array_row', 'array_col'],
                   shape='h',
                   coord_ratio=1,
                   return_plot=False):
    if low:
        plot_df = pd.DataFrame(adata.uns['fms_low'])
        plot_df.index = adata.obs.index
        plot_df.columns = ['low_FM_' + \
                           str(i + 1) for i in range(plot_df.shape[1])]
        if spatial_info in adata.obsm_keys():
            plot_df['x'] = adata.obsm[spatial_info][:, 0]
            plot_df['y'] = adata.obsm[spatial_info][:, 1]
        elif set(spatial_info) <= set(adata.obs.columns):
            plot_coor = adata.obs
            plot_df = plot_df['low_FM_' + str(rank)]
            plot_df = pd.DataFrame(plot_df)
            plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
            plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        plot_df['radius'] = size
        base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y',
                                                        fill='low_FM_' + str(rank)),
                                           shape=shape,
                                           stroke=0.1,
                                           size=size) +
                     xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                     ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                     scale_fill_cmap(cmap_name=cmap) +
                     coord_equal(ratio=coord_ratio) +
                     theme_classic() +
                     theme(legend_position=('right'),
                           legend_background=element_blank(),
                           legend_key_width=4,
                           legend_key_height=50)
                     )
        print(base_plot)

    else:
        plot_df = pd.DataFrame(adata.uns['fms_high'])
        plot_df.index = adata.obs.index
        plot_df.columns = ['high_FM_' + str(i + 1) for i in \
                           range(adata.uns['fms_high'].shape[1])]
        if spatial_info in adata.obsm_keys():
            plot_df['x'] = adata.obsm[spatial_info][:, 0]
            plot_df['y'] = adata.obsm[spatial_info][:, 1]
        elif set(spatial_info) <= set(adata.obs.columns):
            plot_coor = adata.obs
            plot_df = plot_df['high_FM_' + str(plot_df.shape[1] - rank + 1)]
            plot_df = pd.DataFrame(plot_df)
            plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
            plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        plot_df['radius'] = size
        base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y',
                                                        fill='high_FM_' + \
                                                             str(adata.uns['fms_high'].shape[1] - \
                                                                 rank + 1)),
                                           shape=shape, stroke=0.1, size=size) +
                     xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                     ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                     scale_fill_cmap(cmap_name=cmap) +
                     coord_equal(ratio=coord_ratio) +
                     theme_classic() +
                     theme(legend_position=('right'),
                           legend_background=element_blank(),
                           legend_key_width=4,
                           legend_key_height=50)
                     )
        print(base_plot)

    if return_plot:
        return base_plot


def ftu_heatmap_signal_ftu_id_card(adata,
                                   ftu,
                                   ax=None,
                                   title=None):
    freq_signal = \
        adata.uns['detect_ftu_data']['freq_signal_ftu'].loc[ftu,
        :].values.reshape(1, -1)
    # print(freq_signal)
    if title != None:
        plt.title(title, fontsize=10)
    sns.heatmap(data=freq_signal, cbar=False, cmap="Reds")
    ax.tick_params(left=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# ftu id card
def cell_type_proportion_box_ftu_id_card(cell_type_name,
                                         cell_type_proportion_data,
                                         ax=None,
                                         title=None):
    boxplot_data = []
    for val in cell_type_name:
        boxplot_data.append(cell_type_proportion_data[val])
    labels = [x.replace("q05cell_abundance_w_sf_", "") for x in cell_type_name]
    plt.title(title, fontsize=20)
    ax.boxplot(boxplot_data, labels=labels, showfliers=False,
               patch_artist=True,
               boxprops={"facecolor": "#FF2A6B"},
               medianprops={"color": "black"})
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)


# ftu id card
def ftu_spatial_map_scatter_ftu_id_card(adata,
                                        ftu_name,
                                        ftu_color,
                                        title,
                                        shape='h',
                                        radius=0.5,
                                        spatial_info=['array_row', 'array_col'],
                                        ax=None):
    x = []
    y = []
    ftu = list(adata.obsm["ftu_binary"][ftu_name].values)
    ftu = [int(x) for x in ftu]
    cmap = ListedColormap(["lightgray", ftu_color])

    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values

    if title != None:
        plt.title(title, y=-0.2, fontsize=10)
    ax.scatter(y, max(x) - x, s=radius, c=ftu, cmap=cmap, marker=shape)
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.grid(False)


# ftu id card
def ftu_overlapped_scatter_ftu_id_card(adata,
                                       ftu_1_code,
                                       ftu_2_code,
                                       ftu_1_color,
                                       ftu_2_color,
                                       overlapped_color,
                                       title,
                                       marker='h',
                                       radius=0.03,
                                       spatial_info=['array_row', 'array_col'],
                                       ax=None):
    ftu_1 = adata.obsm["ftu_binary"][f"ftu_{ftu_1_code}"].values
    ftu_2 = adata.obsm["ftu_binary"][f"ftu_{ftu_2_code}"].values
    x = []
    y = []

    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    plt.title(title, y=-0.2)

    ftu_x = max(x) - x
    ftu_1_value = []
    ftu_2_value = []
    ftu_overlapped_value = []
    ftu_blank_value = []
    for index in range(len(ftu_1)):
        if ftu_1[index] == ftu_2[index] and ftu_1[index] == "1":
            ftu_overlapped_value.append([y[index], ftu_x[index], 3])
        elif ftu_1[index] == "1":
            ftu_1_value.append([y[index], ftu_x[index], 1])
        elif ftu_2[index] == "1":
            ftu_2_value.append([y[index], ftu_x[index], 2])
        else:
            ftu_blank_value.append([y[index], ftu_x[index], 0])
    label_name = [f"ftu_{ftu_1_code}", f"ftu_{ftu_2_code}", "ftu_overlap", "ftu_blank"]
    ftu_1_value = np.array(ftu_1_value)
    ftu_overlapped_value = np.array(ftu_overlapped_value)
    ftu_2_value = np.array(ftu_2_value)
    ftu_blank_value = np.array(ftu_blank_value)
    scatter_1 = ax.scatter(ftu_1_value[:, 0], ftu_1_value[:, 1],
                           marker=marker, s=radius, c=ftu_1_color)
    scatter_2 = ax.scatter(ftu_2_value[:, 0], ftu_2_value[:, 1],
                           marker=marker, s=radius, c=ftu_2_color)
    scatter_overlap = ax.scatter(ftu_overlapped_value[:, 0],
                                 ftu_overlapped_value[:, 1],
                                 marker=marker,
                                 s=radius,
                                 c=overlapped_color)
    scatter_blank = ax.scatter(ftu_blank_value[:, 0],
                               ftu_blank_value[:, 1],
                               marker=marker,
                               s=radius,
                               c="lightgray")
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.legend([scatter_1], [f"ftu_{ftu_1_code}"])
    ax.legend([scatter_2], [f"ftu_{ftu_2_code}"])
    ax.legend([scatter_overlap], ["ftu_overlap"])
    ax.legend([scatter_blank], ["ftu_blank"])

    plt.legend(label_name, fontsize=5, loc=7, frameon=False,
               bbox_to_anchor=(1, 0, 0.35, 1))
    ax.grid(False)


# ftu id card
def scatter_svgs_distri_ftu_id_card(adata,
                                    gene,
                                    size=None,
                                    shape='h',
                                    cmap='magma',
                                    spatial_info=['array_row', 'array_col'],
                                    ax=None,
                                    coord_ratio=1,
                                    return_plot=False):
    if gene in adata.obs.columns:
        plot_df = pd.DataFrame(adata.obs.loc[:, gene].values,
                               index=adata.obs_names,
                               columns=[gene])
        if spatial_info in adata.obsm_keys():
            plot_df['x'] = adata.obsm[spatial_info][:, 1]
            plot_df['y'] = adata.obsm[spatial_info][:, 0]
        elif set(spatial_info) <= set(adata.obs.columns):
            plot_coor = adata.obs
            plot_df = plot_df[gene]
            plot_df = pd.DataFrame(plot_df)
            plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
            plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        plot_df['radius'] = size
        if size is not None:
            plot_scatter_ftu_id_card(plot_df.y,
                                     max(plot_df.x) - plot_df.x, plot_df[gene],
                                     gene, cmap, radius=plot_df['radius'], ax=ax)
        else:
            plot_scatter_ftu_id_card(plot_df.y,
                                     max(plot_df.x) - plot_df.x, plot_df[gene],
                                     gene, cmap, ax=ax)
        return
    if ss.issparse(adata.X):
        plot_df = pd.DataFrame(adata.X.todense(), index=adata.obs_names,
                               columns=adata.var_names)
    else:
        plot_df = pd.DataFrame(adata.X, index=adata.obs_names,
                               columns=adata.var_names)
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 1]
        plot_df['y'] = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
    plot_df['radius'] = size
    plot_df = plot_df.sort_values(by=gene, ascending=True)
    # print(plot_df)
    if size is not None:
        plot_scatter_ftu_id_card(plot_df.y, max(plot_df.x) - plot_df.x,
                                 plot_df[gene], gene,
                                 cmap, radius=plot_df['radius'], ax=ax)
    else:
        plot_scatter_ftu_id_card(plot_df.y, max(plot_df.x) - plot_df.x,
                                 plot_df[gene], gene,
                                 cmap, ax=ax)


# ftu id card
def ftu_freq_signal_ftu_id_card(adata,
                                ftu,
                                color='#CA1C1C',
                                y_range=[0, 0.08],
                                ax=None, title=None):
    # Show the frequency signal
    freq_signal = \
        adata.uns['detect_ftu_data']['freq_signal_ftu'].loc[ftu, :].values
    low = list(range(len(freq_signal)))
    plt.bar(low, freq_signal, color=color)
    plt.grid(False)
    if title != None:
        plt.title(title, fontsize=20)
    ax.spines['right'].set_color("none")
    ax.spines['top'].set_color("none")
    plt.ylim(y_range[0], y_range[1])
    plt.xlim(0, freq_signal.size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_scatter_ftu_id_card(x, y, colors, title, cmap, shape='h',
                             radius=None, ax=None, up_title=False):
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.9)
    if up_title:
        plt.title(title)
    else:
        plt.title(title, y=-0.2)
    if radius is not None:
        scatter = ax.scatter(x, y, s=radius, c=colors, cmap=cmap, marker=shape)
    else:
        scatter = ax.scatter(x, y, c=colors, cmap=cmap, marker=shape)
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.grid(False)
    return scatter


def scatter_gene(adata,
                 gene,
                 size=None,
                 shape='o',
                 cmap='magma',
                 spatial_info=['array_row', 'array_col'],
                 dpi=100,
                 return_fig=False,
                 save_path=None):
    """
    Visualize genes through given spatial information.

    Parameters
    ----------
    adata : anndata
        spatial datasets. 
    gene : str or list
        The genes which will be visualized.
    size : float, optional
        The size of spots. 
        The default is None.
    shape : str, optional
        The shape of the spots. 
        The default is 'o'.
    cmap : str, optional
        The color theme used. 
        The default is "magma".
    spatial_info : str or list, optional
        The spatial information key in adata.obsm or columns in adata.obs. 
        The default is ['array_row', 'array_col'].
    dpi : int, optional
        DPI. The default is 100.
    return_fig : bool, optional
        Where to return the figure.
        The default is False.
    save_path : system path, optional
        The path for the saving figure.
        The default is None.
    Raises
    ------
    KeyError
        gene should be adata.var_names.

    """
    if isinstance(gene, np.ndarray):
        gene = list(gene)
    if isinstance(gene, pd.core.indexes.base.Index):
        gene = list(gene)
    if ss.issparse(adata.X):
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    else:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 1]
        plot_df['y'] = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        # print(plot_df)
    if isinstance(gene, str):
        fig, ax = plt.subplots()
        if size == None:
            scatter = plot_scatter_ftu_id_card(x=plot_df.y,
                                               y=max(plot_df.x) - plot_df.x,
                                               colors=plot_df[gene],
                                               title=gene,
                                               shape=shape,
                                               cmap=cmap,
                                               ax=ax,
                                               up_title=True)
            plt.colorbar(scatter, ax=ax)
        elif isinstance(size, int) or isinstance(size, float):
            scatter = plot_scatter_ftu_id_card(x=plot_df.y,
                                               y=max(plot_df.x) - plot_df.x,
                                               colors=plot_df[gene],
                                               title=gene,
                                               shape=shape,
                                               cmap=cmap,
                                               radius=size,
                                               ax=ax,
                                               up_title=True)
            plt.colorbar(scatter, ax=ax)

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        if return_fig:
            return ax

    elif isinstance(gene, list) or isinstance(gene, np.ndarray):
        row = math.ceil(len(gene) / 4)
        fig = plt.figure(dpi=dpi,
                         constrained_layout=True,
                         figsize=(20, row * 5))

        gs = GridSpec(row, 4,
                      figure=fig)
        ax_list = []
        for index, value in enumerate(gene):
            ax = fig.add_subplot(gs[index // 4, index % 4])

            if size == None:
                scatter = plot_scatter_ftu_id_card(x=plot_df.y,
                                                   y=max(plot_df.x) - plot_df.x,
                                                   colors=plot_df[value],
                                                   title=value,
                                                   shape=shape,
                                                   cmap=cmap,
                                                   ax=ax,
                                                   up_title=True)
                plt.colorbar(scatter, ax=ax)
            elif isinstance(size, int) or isinstance(size, float):
                scatter = plot_scatter_ftu_id_card(x=plot_df.y,
                                                   y=max(plot_df.x) - plot_df.x,
                                                   colors=plot_df[value],
                                                   title=value,
                                                   shape=shape,
                                                   cmap=cmap,
                                                   radius=size,
                                                   ax=ax,
                                                   up_title=True)
                plt.colorbar(scatter, ax=ax)
            ax_list.append(ax)

        if save_path:
            plt.savefig(save_path)
        plt.show()
        if return_fig:
            return ax_list


def scatter_ftu(adata,
                ftu,
                shape='o',
                ftu_color='#FF6879',
                size=None,
                spatial_info=['array_row', 'array_col'],
                save_path=None,
                return_fig=False):
    """
    Plot the spatial map of a ftu.

    Parameters
    ----------
    adata : anndata
        spatial datasets. 
    ftu : str
        The ftu indicator.
    shape : str, optional
        The shape of spots. The default is 'o'.
    ftu_color : str, optional
        The color of the ftu(s). The default is '#FF6879'.
    size : float, optional
        The size of spots. The default is None.
    spatial_info : str or list, optional
        The spatial information key in adata.obsm or columns in adata.obs. 
        The default is ['array_row', 'array_col'].
    return_fig : bool, optional
        Where to return the figure. The default is False.
    save_path : system path, optional
        The path for the saving figure. The default is None.

    Returns
    -------
    ax_list 
        If return_fig == True, the figure objects will be returned.

    """
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    if isinstance(ftu, str):
        ftu_value = [int(x) for x in list(adata.obsm["ftu_binary"][ftu].values)]
        fig, ax = plt.subplots()
        cmap_ftu = ListedColormap(["#b4b4b4", ftu_color])

        plt.title(ftu)
        if size is not None:
            scatter = ax.scatter(y, max(x) - x, s=size, c=ftu_value,
                                 cmap=cmap_ftu, marker=shape)
        else:
            scatter = ax.scatter(y, max(x) - x, c=ftu_value,
                                 cmap=cmap_ftu, marker=shape)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(labelsize=10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        ax.grid(False)
        plt.legend(*scatter.legend_elements(), loc="center right",
                   bbox_to_anchor=(1, 0, 0.15, 1))

        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax
    elif isinstance(ftu, list) or isinstance(ftu, np.ndarray):
        row = math.ceil(len(ftu) / 4)
        fig = plt.figure(dpi=200,
                         constrained_layout=True,
                         figsize=(20, row * 5)
                         )

        gs = GridSpec(row,
                      4,
                      figure=fig)
        # print(row, 4)
        ax_list = []
        for index, value in enumerate(ftu):
            ax = fig.add_subplot(gs[index // 4, index % 4])
            ftu_value = [int(x) for x in \
                         list(adata.obsm["ftu_binary"][value].values)]
            cmap_ftu = ListedColormap(["#b4b4b4", ftu_color])

            plt.title(value)
            if size is not None:
                scatter = ax.scatter(y, max(x) - x, s=size, c=ftu_value,
                                     cmap=cmap_ftu, marker=shape)
            else:
                scatter = ax.scatter(y, max(x) - x, c=ftu_value,
                                     cmap=cmap_ftu, marker=shape)

            ax.minorticks_on()
            ax.yaxis.set_tick_params(labelsize=10)
            ax.xaxis.set_tick_params(labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            ax.grid(False)
            plt.legend(*scatter.legend_elements(), loc="center right",
                       bbox_to_anchor=(1, 0, 0.15, 1))
            ax_list.append(ax)

        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list


def scatter_ftu_expression(adata,
                           ftu,
                           cmap='magma',
                           shape='o',
                           size=None,
                           spatial_info=['array_row', 'array_col'],
                           save_path=None,
                           return_fig=False):
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    if isinstance(ftu, str):
        ftu_value = adata.obsm["ftu_pseudo_expression"][ftu].values
        fig, ax = plt.subplots()
        plt.title(ftu)
        if size is not None:
            scatter = ax.scatter(y, max(x) - x, s=size, c=ftu_value, cmap=cmap,
                                 marker=shape)
        else:
            scatter = ax.scatter(y, max(x) - x, c=ftu_value, cmap=cmap,
                                 marker=shape)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(labelsize=10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
        ax.grid(False)

        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax
    elif isinstance(ftu, list) or isinstance(ftu, np.ndarray):
        row = math.ceil(len(ftu) / 4)
        fig = plt.figure(dpi=350,
                         constrained_layout=True,
                         figsize=(20, row * 5)
                         )

        gs = GridSpec(row, 4,
                      figure=fig)
        # print(row, 4)
        ax_list = []
        for index, value in enumerate(ftu):
            ax = fig.add_subplot(gs[index // 4, index % 4])
            ftu_value = adata.obsm["ftu_pseudo_expression"][value].values

            plt.title(value)
            if size is not None:
                scatter = ax.scatter(y, max(x) - x, s=size, c=ftu_value,
                                     cmap=cmap, marker=shape)
            else:
                scatter = ax.scatter(y, max(x) - x, c=ftu_value, cmap=cmap,
                                     marker=shape)
            ax.minorticks_on()
            ax.yaxis.set_tick_params(labelsize=10)
            ax.xaxis.set_tick_params(labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax)
            ax.grid(False)
            ax_list.append(ax)

        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list


def overlap_curve(adata, save_path=None, return_fig=False):
    curve_dot = adata.uns['detect_ftu_data']['overlap_curve']
    curve_x = [float(x.replace("res_", "")) for x in list(curve_dot.index)]
    curve_y = list(curve_dot["score"].values)
    x_values = curve_x  # .tolist()
    y_values = curve_y
    ax = plt.scatter(x_values, y_values)
    plt.plot(x_values, y_values)
    plt.title("Overlap curve", fontsize=24)
    plt.xlabel("Resolution", fontsize=14)
    plt.ylabel("Score", fontsize=14)

    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()
    if return_fig:
        return ax


def scatter_umap_clustering(adata,
                            svg_list,
                            size=3,
                            alpha=1,
                            return_fig=False,
                            save_path=None):
    """
    Visualize clustering results.

    Parameters
    ----------
    adata : anndata
        spatial datasets. 
    svg_list : list
        svg list.
    size : float
        The size of genes.
        The default is 3.
    alpha : float, optional
        Transparency.
        The default is 1.
    return_fig : bool, optional
        Where to return the figure. The default is False.
    save_path : system path, optional
        The path for the saving figure. The default is None.

    Raises
    ------
    KeyError
        Run sgagft.identify_tm() before this step.

    Returns
    -------
    base_plot : plotnine object
        The figure.

    """
    current_genes = adata.uns['detect_ftu_data']['gft_umap_ftu'].index.tolist()
    if set(svg_list) <= set(current_genes):
        svg_list = np.intersect1d(svg_list, current_genes)
    else:
        diff_genes = np.setdiff1d(svg_list, current_genes)
        raise KeyError(f"{diff_genes} are not calculated in the above step.")
    plot_df = pd.concat((adata.uns['detect_ftu_data'] \
                             ['gft_umap_ftu'].loc[svg_list, :],
                         adata.var.loc[svg_list, :].ftu), axis=1)

    categories = [eval(i) for i in np.unique(plot_df.ftu)]
    categories = np.sort(np.array(categories))
    categories = categories.astype(str)
    plot_df.ftu = pd.Categorical(plot_df.ftu,
                                 categories=categories)
    base_plot = (ggplot(plot_df,
                        aes('UMAP_1', 'UMAP_2',
                            fill='ftu'))
                 + geom_point(size=size, alpha=alpha, stroke=0.1)
                 + scale_fill_hue(s=0.9, l=0.65, h=0.0417, color_space='husl')
                 + theme_classic())
    print(base_plot)

    if save_path is not None:
        base_plot.save(f"{save_path}")
    if return_fig:
        return base_plot


def scatter_ftu_gene(adata,
                     ftu,
                     gene,  # list
                     shape='o',
                     cmap="magma",
                     ftu_color='#FF6879',
                     size=None,
                     spatial_info=['array_row', 'array_col'],
                     return_fig=False,
                     save_path=None):
    """
    Plot a ftu and several genes simultaneously.

    Parameters
    ----------
    adata : anndata
        spatial datasets. 
    ftu : str
        The ftu indicator.
    gene : list
        The list of gene names.
    shape : str, optional
        The shape of spots.
        The default is 'o'.
    cmap : str, optional
        The color theme used. 
        The default is "magma".
    ftu_color : str, optional
        The color used for representing the ftu. 
        The default is '#FF6879'.
    size : float, optional
        The size of spots. The default is None.
    spatial_info : str or list, optional
        The spatial information key in adata.obsm or columns in adata.obs. 
        The default is ['array_row', 'array_col'].
    return_fig : bool, optional
        Where to return the figure. The default is False.
    save_path : system path, optional
        The path for the saving figure. The default is None.
    Raises
    ------
    KeyError
        gene should be found at adata.var_names.

    Returns
    -------
    ax_list : matplotlist subaxes object list
        The figures.

    """

    if isinstance(gene, str):
        gene = [gene]
    if isinstance(gene, pd.core.indexes.base.Index):
        gene = list(gene)
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    row = math.ceil((1 + len(gene)) / 4)
    fig = plt.figure(dpi=350,
                     constrained_layout=True,
                     figsize=(20, row * 5)
                     )

    gs = GridSpec(row, 4,
                  figure=fig)
    ax_list = []
    ###########################################################
    ax_ftu = fig.add_subplot(gs[0, 0])
    ftu_value = [int(x) for x in list(adata.obsm["ftu_binary"][ftu].values)]
    cmap_ftu = ListedColormap(["#b4b4b4", ftu_color])

    plt.title(ftu)
    if size is not None:
        scatter = ax_ftu.scatter(y, max(x) - x, s=size, c=ftu_value, cmap=cmap_ftu)
    else:
        scatter = ax_ftu.scatter(y, max(x) - x, c=ftu_value, cmap=cmap_ftu)

    ax_ftu.minorticks_on()
    ax_ftu.yaxis.set_tick_params(labelsize=10)
    ax_ftu.xaxis.set_tick_params(labelsize=10)
    ax_ftu.spines['top'].set_visible(False)
    ax_ftu.spines['right'].set_visible(False)
    ax_ftu.spines['left'].set_visible(False)
    ax_ftu.spines['bottom'].set_visible(False)
    ax_ftu.get_xaxis().set_visible(False)
    ax_ftu.get_yaxis().set_visible(False)
    ax_ftu.set_aspect('equal')
    ax_ftu.grid(False)
    plt.legend(*scatter.legend_elements(), loc="center right",
               bbox_to_anchor=(1, 0, 0.15, 1))
    ax_list.append(ax_ftu)
    #########################

    if isinstance(gene, np.ndarray):
        gene = list(gene)
    if ss.issparse(adata.X):
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    else:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 1]
        plot_df['y'] = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        # print(plot_df)
    if isinstance(gene, list) or isinstance(gene, np.ndarray):
        for index, value in enumerate(gene):
            ax = fig.add_subplot(gs[(1 + index) // 4, (1 + index) % 4])

            if size == None:
                scatter = plot_scatter_ftu_id_card(x=plot_df.y,
                                                   y=max(plot_df.x) - plot_df.x,
                                                   shape=shape,
                                                   colors=plot_df[value],
                                                   title=value,
                                                   cmap=cmap,
                                                   ax=ax,
                                                   up_title=True)
            elif isinstance(size, int) or isinstance(size, float):
                scatter = plot_scatter_ftu_id_card(x=plot_df.y,
                                                   y=max(plot_df.x) - plot_df.x,
                                                   colors=plot_df[value],
                                                   title=value,
                                                   cmap=cmap,
                                                   radius=size,
                                                   ax=ax,
                                                   up_title=True)
            plt.colorbar(scatter, ax=ax)
            ax_list.append(ax)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    if return_fig:
        return ax_list


def draw_ftu_id_card(adata,
                     svg_list,
                     ftu,
                     deconvolution_key='cell_type_proportion',
                     spatial_info=['array_row', 'array_col'],
                     shape='h',
                     dpi=350,
                     size=[7, 0.8],
                     return_fig=False,
                     save_path=None):
    """
    Plot the details of a ftu to generate a ftu ID CARD.

    Parameters
    ----------
    adata : anndata
        spatial datasets. 
    svg_list : list
        svg list.
    ftu : str
        The ftu indicator.
    deconvolution_key : str, optional
        The deconvolution results should be found at 
        adata.obsm[deconvolution_key]. 
        The default is 'cell_type_proportion'.
    spatial_info : str or list, optional
        The spatial information key in adata.obsm or columns in adata.obs. 
        The default is ['array_row', 'array_col'].
    shape : str
        The shape of the spots.
        The default is 'h'.
    dpi : int, optional
        Dots per inch. The default is 350.
    size : float list, optional
        Note there are two sizes of spots, corresponding to the large figures 
        and the small figures.
        The default is [7, 0.8].
    return_fig : bool, optional
        Where to return the figure. The default is False.
    save_path : system path, optional
        The path for the saving figure. The default is None.

    Returns
    -------
    fig : matplotlib axes
        Figure.

    """
    if 'ftu-' in ftu:
        ftu = ftu.replace('ftu-', '')
    gene_df = adata.var.loc[svg_list, :]

    ftu_total = [str(ind) for ind in range(1,
                                           1 + np.unique(gene_df['ftu']).size)]
    fig = plt.figure(dpi=dpi,
                     constrained_layout=True,
                     figsize=(12, 14))
    gene_df = gene_df.loc[svg_list, :]
    ftu_gene_list = gene_df.loc[gene_df['ftu'] == ftu,
                    :].index.tolist()

    # *************************************************
    # Spatial map plot scatter
    ax_SpaMap_scatter = plt.subplot2grid((12, 14), (1, 1), colspan=4,
                                         rowspan=4)
    ftu_spatial_map_scatter_ftu_id_card(adata,
                                        f"ftu_{ftu}",
                                        "#FF6879",
                                        radius=size[0],
                                        title=None,
                                        ax=ax_SpaMap_scatter,
                                        spatial_info=spatial_info,
                                        shape=shape,
                                        )
    # Spatial map plot start
    # Spatial map title
    ax_SpaMap_title = plt.subplot2grid((12, 14), (0, 1), colspan=4, rowspan=1)
    plt.title(f"Spatial map: ftu {ftu}", y=0, fontsize=20)
    ax_SpaMap_title.spines['top'].set_visible(False)
    ax_SpaMap_title.spines['right'].set_visible(False)
    ax_SpaMap_title.spines['left'].set_visible(False)
    ax_SpaMap_title.spines['bottom'].set_visible(False)
    ax_SpaMap_title.get_xaxis().set_visible(False)
    ax_SpaMap_title.get_yaxis().set_visible(False)

    # Spatial map plot end
    # *************************************************

    # *************************************************
    # Enhanced svgs plot start
    # Enhanced svgs title
    ax_enhsvgs_title = plt.subplot2grid((12, 14), (0, 6), colspan=4, rowspan=1)

    plt.title("Corresponding svgs", y=0, fontsize=20)
    ax_enhsvgs_title.spines['top'].set_visible(False)
    ax_enhsvgs_title.spines['right'].set_visible(False)
    ax_enhsvgs_title.spines['left'].set_visible(False)
    ax_enhsvgs_title.spines['bottom'].set_visible(False)
    ax_enhsvgs_title.get_xaxis().set_visible(False)
    ax_enhsvgs_title.get_yaxis().set_visible(False)
    # Enhanced svgs plot scatters
    for index in range(min(8, len(ftu_gene_list))):
        ax = plt.subplot2grid((12, 14), (1 + 2 * (index % 4),
                                         6 + 2 * (index // 4)),
                              colspan=2, rowspan=2)

        scatter_svgs_distri_ftu_id_card(adata,
                                        gene=ftu_gene_list[index],
                                        spatial_info=spatial_info,
                                        size=size[1],
                                        shape=shape,
                                        ax=ax)

    # Enhanced svgs plot end
    # *************************************************

    # *************************************************
    # Overlapped ftus plot start
    ftu_overlapper_max = {
        "ftu": None,
        "ob_sum": None
    }
    ftu_overlapper_min = {
        "ftu": None,
        "ob_sum": None
    }
    for value in ftu_total:
        v1 = [int(i) for i in adata.obsm["ftu_binary"][f"ftu_{value}"].values]
        v2 = [int(i) for i in adata.obsm["ftu_binary"][f"ftu_{ftu}"].values]
        current_value = [i for i in list(map(lambda x: x[0] + x[1],
                                             zip(v2, v1))) if i == 2]
        current_value = len(current_value)
        if ftu_overlapper_max["ftu"] == None and value != ftu:
            ftu_overlapper_max["ftu"] = value
            ftu_overlapper_max["ob_sum"] = current_value

        elif ftu_overlapper_min["ftu"] == None and value != ftu:
            ftu_overlapper_min["ftu"] = value
            ftu_overlapper_min["ob_sum"] = current_value
        elif value != ftu:
            if ftu_overlapper_min["ob_sum"] > current_value:
                ftu_overlapper_min["ob_sum"] = current_value
                ftu_overlapper_min["ftu"] = value
            if ftu_overlapper_max["ob_sum"] < current_value:
                ftu_overlapper_max["ob_sum"] = current_value
                ftu_overlapper_max["ftu"] = value
    ax_Overlapped_scatter_1 = plt.subplot2grid((12, 14), (5, 1),
                                               colspan=2,
                                               rowspan=2)
    ftu_overlapped_scatter_ftu_id_card(adata,
                                       ftu,
                                       ftu_overlapper_max["ftu"],
                                       ftu_1_color="#FF6879",
                                       ftu_2_color="Green",
                                       overlapped_color="Yellow",
                                       title="Overlapped ftus",
                                       radius=size[1],
                                       spatial_info=spatial_info,
                                       marker=shape,
                                       ax=ax_Overlapped_scatter_1)
    ax_Overlapped_scatter_2 = plt.subplot2grid((12, 14), (7, 1),
                                               colspan=2, rowspan=2)
    ftu_overlapped_scatter_ftu_id_card(adata,
                                       ftu,
                                       ftu_overlapper_min["ftu"],
                                       ftu_1_color="#FF6879",
                                       ftu_2_color="Green",
                                       overlapped_color="Yellow",
                                       title="Overlapped ftus",
                                       radius=size[1],
                                       marker=shape,
                                       spatial_info=spatial_info,
                                       ax=ax_Overlapped_scatter_2)
    ax_Overlapped_scatter_1_ftu = plt.subplot2grid((12, 14), (5, 3),
                                                   colspan=2, rowspan=2)

    ftu_spatial_map_scatter_ftu_id_card(adata,
                                        f"ftu_{str(ftu_overlapper_max['ftu'])}",
                                        "#FF6879",
                                        f"ftu {ftu_overlapper_max['ftu']}",
                                        ax=ax_Overlapped_scatter_1_ftu,
                                        radius=size[1],
                                        shape=shape,
                                        spatial_info=spatial_info)
    ax_Overlapped_scatter_2_ftu = plt.subplot2grid((12, 14), (7, 3),
                                                   colspan=2, rowspan=2)

    ftu_spatial_map_scatter_ftu_id_card(adata,
                                        f"ftu_{str(ftu_overlapper_min['ftu'])}",
                                        "#FF6879",
                                        f"ftu {ftu_overlapper_min['ftu']}",
                                        ax=ax_Overlapped_scatter_2_ftu,
                                        radius=size[1],
                                        shape=shape,
                                        spatial_info=spatial_info)
    # Overlapped ftus plot end
    # *************************************************

    # *************************************************
    # Cell type proportion plot start
    # Cell type proportion title
    cell_type_proportion_plot = plt.subplot2grid((12, 14), (7, 6),
                                                 colspan=4, rowspan=2)

    # Cell type proportion plot
    if deconvolution_key is not None:
        cell2loc = adata.obsm[deconvolution_key]
        sum_cell2loc = [sum(cell2loc[i].values.tolist()) for i in \
                        cell2loc.columns.tolist()]
        # print(sum_cell2loc)
        cell_type_name_i = [sum_cell2loc.index(x) for x in \
                            sorted(sum_cell2loc, reverse=True)[:10]]
        cell_type_name = [cell2loc.columns.tolist()[i] for i \
                          in cell_type_name_i]

        cell_type_proportion_box_ftu_id_card(cell_type_name,
                                             cell2loc,
                                             ax=cell_type_proportion_plot,
                                             title=deconvolution_key)
    # Cell type proportion plot end
    # *************************************************

    # *************************************************
    # svg functional enrichment plot start
    import gseapy as gp
    enr = gp.enrichr(gene_list=ftu_gene_list,
                     gene_sets=['BioPlanet_2019',
                                'GO_Biological_Process_2021',
                                'ChEA_2016'],
                     organism='Human',
                     description='ftu',
                     outdir='ftup/enrichr_kegg',
                     no_plot=False,
                     cutoff=0.5
                     )

    GO_Biological_Processn_plot = plt.subplot2grid((12, 14), (10, 1),
                                                   colspan=4, rowspan=2)

    from gseapy.plot import barplot
    GO_Biological_Process_file = "./ftup/enrichr_kegg/GO_Biological_Process"
    barplot(enr.results[enr.results.Gene_set == 'GO_Biological_Process_2021'],
            top_term=10, ofname=GO_Biological_Process_file, )
    from PIL import Image
    img = Image.open(GO_Biological_Process_file + ".png", )
    GO_Biological_Processn_plot.imshow(img)
    plt.title("GO Biological Process", fontsize=20)
    GO_Biological_Processn_plot.spines['top'].set_visible(False)
    GO_Biological_Processn_plot.spines['right'].set_visible(False)
    GO_Biological_Processn_plot.spines['left'].set_visible(False)
    GO_Biological_Processn_plot.spines['bottom'].set_visible(False)
    GO_Biological_Processn_plot.get_xaxis().set_visible(False)
    GO_Biological_Processn_plot.get_yaxis().set_visible(False)
    # svg functional enrichment's plot end
    # *************************************************

    # *************************************************

    ftu_heatmap_signal_plot = plt.subplot2grid((12, 14), (11, 6),
                                               colspan=2, rowspan=1)

    ftu_heatmap_signal_ftu_id_card(adata,
                                   ftu=f"ftu_{ftu}",
                                   ax=ftu_heatmap_signal_plot)

    ftu_freq_signal_plot = plt.subplot2grid((12, 14), (10, 6),
                                            colspan=2, rowspan=1)

    ftu_freq_signal_ftu_id_card(adata,
                                ftu=f"ftu_{ftu}",
                                ax=ftu_freq_signal_plot,
                                title="FCs")
    logo_plot = plt.subplot2grid((12, 14), (10, 8), colspan=2, rowspan=2)

    # Logo_plot_file = "./SpaGFT_Logo_RGB.png"
    # from PIL import Image
    # img = Image.open(Logo_plot_file)
    # Logo_plot.imshow(img)
    logo_plot.spines['top'].set_visible(False)
    logo_plot.spines['right'].set_visible(False)
    logo_plot.spines['left'].set_visible(False)
    logo_plot.spines['bottom'].set_visible(False)
    logo_plot.get_xaxis().set_visible(False)
    logo_plot.get_yaxis().set_visible(False)
    plt.text(x=0, y=0.5, s="Powered \n      by  \n SpaGFT",
             fontsize=20, fontstyle="italic",
             fontweight="bold")
    plt.title("ftu " + ftu + " ID Card \n# of svgs: " + \
              str(len(ftu_gene_list)), y=0)
    # # *************************************************
    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}/ftu_{ftu}.png")
    if os.path.exists("./ftup/enrichr_kegg"):
        os.system("rm -r ./ftup/enrichr_kegg")
    if return_fig:
        return fig
    plt.show()
    plt.close()

