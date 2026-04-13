import time
import warnings
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy import stats
from scipy.stats import chi2, entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm, trange
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import davies_bouldin_score as dbs
from sklearn.metrics import silhouette_score as sil
from collections import defaultdict
from itertools import combinations_with_replacement
import os
import pickle

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array
    warnings.warn("scipy.sparse.csr_array is not available. Recommend to install scipy >= 1.8")


class Distribution(defaultdict):
    """Distribution used to model the gene expression.
    """
    def __init__(self, value=None, prob=None) -> None:
        super().__init__(float)
        if (value is not None) and (prob is not None):
            if not isinstance(value, (list, np.ndarray)) or \
                not isinstance(prob, (list, np.ndarray)
            ):
                value = list(value)
                prob = list(prob)

            for i, j in zip(value, prob):
                self[i] = j

    @property
    def counts(self):
        return np.array(list(super().keys()))

    @property
    def prob(self):
        return np.array(list(super().values()))

    def prob_check(self):
        assert self.prob.sum() == 1.0, "Probability does not sum to 1.0"

    def from_obs(self, obs):
        ind, count = np.unique(obs, return_counts=True)
        self.__init__(value=ind, prob=count / count.sum())
        return self

    def dist_abs_subtract(self):
        new_dist = Distribution()
        counts = self.counts
        prob = self.prob
        for i, j in combinations_with_replacement(range(len(self)), 2):
            new_dist[np.abs(counts[i] - counts[j])] += (
                prob[i] * prob[j] * (2 - (i == j))
            )
        return new_dist

    def __getitem__(self, __k):
        # if __k is a iterable
        if isinstance(__k, (list, tuple, np.ndarray)):
            return np.array([self[i] for i in __k])
        # if __k is a scalar
        else:
            return super().__getitem__(__k)

    def isf(self, n, alpha=0.05):
        try:
            return np.where(np.cumsum(self[np.arange(n)]) > (1 - alpha))[0][0] + 1
        except:
            return n+1


def adata_preprocess(anndata,copy=True):
    """Preprocess the AnnData object.

    Args:
        anndata (AnnData): AnnData object.
        copy (bool, optional): Defaults to True.

    Returns:
        AnnData: preprocessed AnnData object.
    """
    if copy:
        anndata = anndata.copy()

    sc.pp.normalize_per_cell(anndata, counts_per_cell_after=10000)
    sc.pp.log1p(anndata)
    sc.pp.scale(anndata)
    # anndata.X = (anndata.X - anndata.X.mean(0)) / anndata.X.std(0)
    return anndata


def adata_preprocess_int(anndata,eps = 1e-7,exclude_highly_expressed=True,copy=True):
    """Special preprocess the AnnData object to convert the data to int. This process will keep the median of gene expression after the normalization.

    Args:
        anndata (AnnData): AnnData object.
        eps (float, optional): Used to escape the devide 0 error. Defaults to 1e-7.
        exclude_highly_expressed (bool, optional): Exclude the HVG or not in `sc.pp.normalize_total`. Defaults to True.
        copy (bool, optional): Defaults to True.

    Returns:
        AnnData: preprocessed AnnData object.
    """
    if copy:
        adata = anndata.copy()

    sc.pp.normalize_total(
        adata,
        exclude_highly_expressed=exclude_highly_expressed,
        )
    sc.pp.log1p(adata)
    expr_median = eps+np.median(adata.X,axis=0)
    adata.X = ((adata.X / expr_median) * expr_median).astype(int)

    return adata


def elbow(X: np.ndarray) -> int:
    """Elbow method to find the optimal number of clusters.

    Args:
        X (np.ndarray): Metrics array.

    Returns:
        int: The optimal number of clusters.
    """
    max_idx = np.argmax(X)

    X = X[max_idx:]  # truncate data from max (not origin) to endpoint.
    b = np.array([len(X), X[-1] - X[0]])  # Vector from origin to end.
    norm_vec = [0]  # Initial point ignored.

    for i in range(1, len(X)):
        p = np.array([i, X[i] - X[0]])  # Vector from origin to current point on curve.
        d = np.linalg.norm(p - (np.dot(p, b) / np.dot(b, b)) * b)  # Distance from point to b.

        norm_vec.append(d)

    # Pick the longest connecting line - note max_idx added to slice back into original data.
    return max_idx + np.argmax(norm_vec) 


def modularity(X, labels):
    """Modularity metrics.

    Args:
        X (np.ndarray): Affinity matrix. (n_sample, n_sample)
        labels (list like): Labels of the nodes. (n_sample, )

    Returns:
        float: Modularity value.
    """
    # convert the affinity matrix to an adjacency matrix
    adj = (X > 0).astype(int)
    m = adj.sum() / 2
    k = adj.sum(axis=1)
    # comms is a list of arrays, where each array contains the indices of the nodes in the community
    comms = [np.where(labels == i)[0] for i in np.unique(labels)]
    Q = 0
    for comm in comms:
        # get the submatrix of the adjacency matrix corresponding to the community
        sub_adj = adj[np.ix_(comm, comm)]
        # get the subvector of the degree vector corresponding to the community
        sub_k = k[comm]
        # compute the modularity contribution of the community
        Q += (sub_adj.sum() / (2 * m)) - ((sub_k.sum() / (2 * m)) ** 2)
    # return the modularity value
    return Q


class AutoCluster():
    def __init__(self,criteria='inertia',init_k = 3,max_k=10) -> None:
        
        self.max_k = max_k
        self.init_k = init_k
        criteria_funcs = {
            'bic': self.bic,
            'inertia': self.inertia,
            'sh': sil,
            'ch': chs,
            'db': dbs,
        }
        self.criteria = criteria

        if isinstance(criteria,str):
            self.criteria_func = criteria_funcs[criteria]
        else:
            self.criteria_func = criteria


    def bic(self,model,X,labels):
        n_params = len(np.unique(labels)) * (X.shape[1] + 1)
        return -2 * model.score(X) * X.shape[0] + n_params * np.log(X.shape[0])


    def inertia(self,model,X=None,labels=None):
        return model.inertia_


    def fit(self,X,model=None,verbose=False,**kwargs):
        if model is None and X.shape[0] > 10000:
            self.model = MiniBatchKMeans
        elif model is None and X.shape[0] <= 10000:
            self.model = KMeans
        else:
            self.model = model
        if verbose:
            print(f'Sample size: {X.shape[0]}, using model: {self.model.__name__}')

        self.scores = []
        for k in trange(self.init_k,self.max_k+1,disable=not verbose, desc='finding best cluster number'):
            cluster = self.model(n_clusters=k,**kwargs)
            cluster.fit(X)
            if self.criteria in ['bic','inertia']:
                self.scores.append(
                    self.criteria_func(
                        model=cluster,labels=cluster.labels_,X=X))
            else:
                self.scores.append(
                    self.criteria_func(
                        X=X,labels=cluster.labels_))
        self.scores = np.array(self.scores)
        if self.criteria in ['bic','inertia']:
            self.best_k = elbow(self.scores) + self.init_k
        else:
            self.best_k = np.argmax(self.scores) + self.init_k


    def predict(self,X,**kwargs):
        cluster = self.model(n_clusters=self.best_k,**kwargs)
        cluster.fit(X)
        return cluster.labels_
    
    
    def fit_predict(self,X,model=None,verbose=False,**kwargs):
        self.fit(X,model=model,verbose=verbose,**kwargs)
        if verbose:
            print(f'Best k: {self.best_k}, Now predicting')
        return self.predict(X,**kwargs)


    def plot_elbow(self,ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(range(self.init_k,self.max_k+1),self.scores)
        # show best k
        ax.plot(self.best_k,self.scores[self.best_k-self.init_k],'o',color='red')
        ax.text(self.best_k,self.scores[self.best_k-self.init_k],f'Best k: {self.best_k}')
        ax.set_xlabel('k')
        ax.set_ylabel(self.criteria)
        ax.set_title(f'Elbow for {self.criteria}')
        return ax
    

class Spanve(object):
    """Spanve model.

    key attributes:
    Input:
        - adata: AnnData object.
        - X: expression data.
        - L: spatial information.
        - K: number of neighbors.
        - n_jobs: parallel jobs.
        - hypoth_type: hypothesis of gene expression distribution.
        - verbose: verbose.
    Output:
        - *result_df: result dataframe.
        - rejects: rejected genes.
        - fdrs: false discovery rate.
        - pvals: p-value.
        - overall_dist: overall distribution. (observed distribution of genes)
        - overall_max: overall max expression. (help to determine the freedom of the distribution)
    """
    def __init__(
        self, adata,
        spatial_info=None,
        layer = None,
        neighbor_finder=None,
        K: int = None, 
        hypoth_type: str = "nodist",        
        n_jobs: int = -1, 
        verbose:bool=False,
        **kwargs
    ) -> None:
        """Initialize the Spanve model.

        Args:
            adata (AnnData): AnnData object.
            K (int, optional): Number of K neighbors. Defaults to None, which means using `adata.shape[0]//100` or 5. With larger K, the Spanve is more specific while low sensitivity, and vice versa.
            spatial_info (, optional): Information of spatial location. If a `str` type, it should be in `obsm` keys. If is a `pd.DataFrame` or `np.ndarray`, it should have a length of number of samples. Defaults to None ("spatial").
            layer (str, optional): used adata layers to computated. Defaults to None (`adata.X`).
            neighbor_finder (str, optional): spatial graph construction method, one of ['knn','Delaunay']. Defaults to None ('knn').
            hypoth_type (str, optional): Hypothesis of gene expression distribution. Defaults to "nodist" (No distribution hypothesis). or 'possion'
            n_jobs (int, optional): Parallel jobs. Defaults to -1 to use all kernel.
            verbose (bool, optional): Defaults to False.

        Raises:
            TypeError: if `spatial_info` is not valid.
        """
        super().__init__()
        n_genes = adata.shape[1]
        sc.pp.filter_genes(adata,min_counts=1)
        if adata.shape[1] < n_genes:
            print(f'Filter genes with min_counts=1, {n_genes-adata.shape[1]} genes removed.')
        self.adata = adata
        if K is None and neighbor_finder == 'knn':
            warnings.warn(
                "WARNNING: K is not defined, will use `adata.shape[0]//100` as K."
                "\n----------------------------------------"
                "\n`K` is an important parameter of Spanve model, refer to the number of neibors when building KNN network. With larger K, the Spanve is more specific while low sensitivity, and vice versa."
                "you can set `K` to a proper value based on the number of cells and the number of spatial variably genes."
            )
        self.K = min(max(K if K is not None else self.adata.shape[0]//100, 5), 100)
        self.n_jobs = n_jobs
        self.hypoth_type = hypoth_type
        self.verbose = verbose
        
        if neighbor_finder is None:
            if self.adata.shape[0] < 10000:
                self.neighbor_finder = "knn"
            else:
                warnings.warn(
                    "WARNNING: The number of cells is large, will use `Delaunay` as neighbor_finder."
                    "\n----------------------------------------"
                    "\n`Delaunay` is a spatial graph construction method, which is more suitable for large number of cells."
                    "you can set `neighbor_finder` to `knn` or `Delaunay` based on the number of cells."
                )
                self.neighbor_finder = "Delaunay"
        else:
            self.neighbor_finder = neighbor_finder

        if layer is not None:
            X = adata.layers[layer]
        else:
            X = adata.X.astype(int)
        if issparse(X):
            X = X.toarray()
        if spatial_info is None:
            assert 'spatial' in adata.obsm.keys(), "'spatial' is not in obsm keys, try set param `spatial_info`" 
            L = adata.obsm["spatial"]
        elif isinstance(spatial_info, str):
            L = adata.obsm[spatial_info]
        elif isinstance(spatial_info, np.ndarray):
            L = spatial_info
        elif isinstance(spatial_info, pd.DataFrame):
            L = spatial_info.loc[adata.obs_names,:].values
        else:
            raise TypeError(f'spatial_info is not valid. Now get type {type(spatial_info)}; spatial_info can be str[key of obsm], numpy.ndarry and pd.DataFrame.')

        self.X = X
        self.L = L

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__input_check(verbose=verbose)


    def __input_check(self,verbose=False):
        if not hasattr(self.adata, "X"):
            raise ValueError("adata.X is not defined")
        
        assert self.neighbor_finder in ["knn","Delaunay"], f"neighbor_finder should be 'knn' or 'Delaunay', now get {self.neighbor_finder}"
        assert self.hypoth_type in ["nodist","possion"], f"hypoth_type should be 'nodist' or 'possion', now get {self.hypoth_type}"
        assert self.X.shape[0] == self.L.shape[0], f"expression data is not consistent with spatial data, now get {self.X.shape[0]} != {self.L.shape[0]}"

        if self.adata.X.dtype not in [np.int64, np.int32, np.int16, np.int8]:
            warnings.warn("""
            WARNNING: X must be an int matrix; 
            ----------------------------------------
            Will NOT automatically convert to int. Inputs can be Raw Counts or use `adata_preprocess_int` to get a normalized data with int dtype. """
            )


    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


    def spatial_colocate_preprocess(self, search_space, groupby=None, verbose=False, copy = False):
        """spatial colocate preprocess.

        Args:
            search_space (list): list of gene pairs. 
            groupby (str, optional): calculate the co-locate based on the groups' mean and std. Defaults to None (Not used).
            verbose (bool, optional): Defaults to False.
            copy (bool, optional): Defaults to False.

        Returns:
            if `copy` is True, return the modified Spanve object. Otherwise, modify the original object. 
        """
        if copy:
            self = self.copy()
        else:
            warnings.warn("""
                WARNNING: This function will replace the input data with a new AnnData object.
                ----------------------------------------
                The original data is stored in `self.raw_data`.
            """)


        def colocate_single(x,y):
            sample_corr = (x - x.mean()) * (y - y.mean()) / (x.std() * y.std()+1e-7)
            return sample_corr.astype(int)


        def colocate_group(adata):
            newdf = pd.DataFrame(
                colocate_single(adata[:,list(var1)].X.toarray(),adata[:,list(var2)].X.toarray()),
                index = adata.obs_names,
                columns = [f"{i}~{j}" for i,j in search_space]
            )

            newad = sc.AnnData(newdf,obsm={'spatial':adata.obsm['spatial']},dtype=int)
            return newad

        
        new_search_space = list(set(search_space))
        if len(new_search_space) != len(search_space) and verbose:
            print('There are duplicated gene pairs in search_space, will remove the duplicated pairs.')
        search_space = new_search_space
        
        adata = self.adata.copy()
        self.raw_data = adata
        var1,var2 = zip(*search_space)

        if groupby is not None:
            assert groupby in adata.obs.columns, "groupby should be obs columns and should be categories." 
            groups = adata.obs_vector(groupby)
            n_groups = np.unique(groups).size
            newads = []

            bar = tqdm(total=n_groups,disable=not verbose, desc = f'There are {n_groups} groups. Will cal coexp strength separately.') 
            for g in np.unique(groups):
                adata_ = adata[groups==g,:]
                newad_ = colocate_group(adata_)
                newads.append(newad_)
                bar.update(1)
            newad = sc.concat(newads)
            bar.close()
        else:
            newad = colocate_group(adata)

        sc.pp.filter_genes(newad,min_counts=1)
        # show the number of genes pair filtered
        print(
            "Delete gene pair with pseduo counts < 1:", newad.shape[1]-self.adata.shape[1],"pairs",
            "\nRemaining:",newad.shape[1],"pairs")
        self.adata = newad
        self.X = newad.X
        if copy:
            return self


    def spatial_coexp_coeff(self,search_genes=None,verbose=None, return_vague_p = False):
        """Spatial co-expression coefficient. This function calculates a spatial affinity matrix modified Pearson correlation coefficient. For the details, please refer to the original paper. The vague p-value is calculated based on the t-distribution, and the effective sample size is calculated based on the number of neighbors. The vague p-value is not strictly correct, but it can be used as a reference.

        Args:
            search_genes (list, optional): list of genes. Defaults to None, which means using `self.adata.var_names[self.rejects]`.
            verbose (optional): Defaults to None.
            return_vague_p (bool, optional): if return a vague p-value or not. Defaults to False.

        Returns:
            if `return_vague_p` is False, return the spatial co-expression coefficient. Otherwise, return the spatial co-expression coefficient and the vague p-value.
        """
        verbose = self.verbose if verbose is None else verbose
        if search_genes is None:
            assert hasattr(self,'rejects'), 'rejects is not defined, please run `fit` first or set `search_genes`'
        search_genes = self.adata.var_names[self.rejects] if search_genes is None else search_genes
        search_genes_idx = self.adata.var_names.isin(search_genes)

        nbr_indices = self.finding_spatial_neibors(K=self.K)
        # transform to csr matrix
        nbr_matrix = csr_array(
            (np.ones(len(nbr_indices[0])),nbr_indices),
            shape=(self.X.shape[0],self.X.shape[0])
        )
        X = self.X[:,search_genes_idx].copy()
        X = X - X.mean(axis=0)

        cov = X.T @ nbr_matrix @ X
        var = np.sqrt(np.diag(cov).reshape(-1, 1) * np.diag(cov).reshape(1, -1))
        cor = np.array(cov / var)
        if not return_vague_p:
            return cor

        n_eff = nbr_matrix.shape[0] / self.K
        assert n_eff > 3, '`self.K` is too large, thus making effective sample size too small.'
        t = cor * np.sqrt(n_eff - 2) / np.sqrt(1 - cor**2)
        p = stats.t.sf(np.abs(t), n_eff-2)*2
        return cor, p


    def build_gene_graph_from_svgenes(
        self, search_genes=None, aff_thres=None, verbose=None):
        """build gene graph from spatial genes.

        Args:
            search_genes (list, optional): The gene list. Defaults to None, which means using `self.adata.var_names[self.rejects]`.
            aff_thres (float, optional): the threshold set to the edge weight. Defaults to None. If None, the threshold will be set to the value where the frequency of the spatial correlation is the highest.
            verbose (optional): Defaults to None.

        Returns:
            np.ndarry: the affinity matrix of the graph.
        """

        verbose = self.verbose if verbose is None else verbose
        search_genes = self.adata.var_names[self.rejects] if search_genes is None else search_genes

        spacor, ps = self.spatial_coexp_coeff(search_genes,verbose,True)
        fdrs = multipletests(ps[np.tril_indices_from(ps,-1)],method='fdr_bh')[1]
        fdrs_ = np.ones_like(ps)
        fdrs_[np.tril_indices_from(ps,-1)] = fdrs

        freq, x_cor, f = plt.hist(
            spacor[np.tril_indices_from(spacor,k=-1)],
            bins=100, density=True, alpha=0.5
        )
        aff_thres = x_cor[np.diff(freq).argmax()] if aff_thres is None else aff_thres

        plt.ylabel('Frequency')
        plt.xlabel('Spatial correlation')
        plt.title('Spatial correlation distribution')
        plt.text(
            aff_thres + 0.05, 
            freq.max() / 10, 
            f'Affinity threshold: {aff_thres:.2f}',
            rotation=90, fontdict=dict(color = 'r',)
        )
        plt.vlines(aff_thres, 0, freq.max(), color='r', linestyle='--')
        if not verbose: 
            plt.close()

        aff = spacor.copy()
        aff[fdrs_ > 0.05] = 0
        aff[np.diag_indices_from(aff)] = 0
        aff[aff < aff_thres] = 0
        return aff


    def detect_spatial_pattern(
        self, clustering_method = 'networkx', # or sklearn
        n_clusters = None, # list, None, or int
        search_genes=None, aff_thres=None,
        verbose=None, seed = 2233,
        **kwargs
    ):
        """detect spatial pattern from SV genes.

        Args:
            clustering_method (str, optional): 'networkx' (Louvain clustering) or 'sklearn' (Spectrum clustering). Defaults to 'networkx'.
            n_clusters (list, None, or int, optional):  Only used when `clustering_method` is 'sklearn'. Defaults to None.
                if None, will search cluster number from 3 to 10. if list, will use `AutoCluster` to search the best cluster number, with format [init_k, max_k].
            search_genes (list, optional): genes list. Defaults to None.
            aff_thres (float, optional): the threshold set to the edge weight. Defaults to None. If None, the threshold will be set to the value where the frequency of the spatial correlation is the highest.
            verbose (optional): Defaults to None.
            seed (int, optional): random seed. Defaults to 2233.

        Raises:
            ValueError: clustering_method should be `sklearn` or `networkx`, or custom graph clustering genes after running `build_gene_graph_from_svgenes`.

        Returns:
            np.ndarray: labels of the spatial pattern.
        """
        np.random.seed(seed)
        if search_genes is None:
            assert hasattr(self,'rejects'), 'rejects is not defined, please run `fit` first or set `search_genes`'
        search_genes = self.adata.var_names[self.rejects] if search_genes is None else search_genes
        aff = self.build_gene_graph_from_svgenes(search_genes,aff_thres,verbose)
        if clustering_method == 'sklearn':
            from sklearn.cluster import SpectralClustering
            if n_clusters is None or isinstance(n_clusters, list):
                if n_clusters is None: 
                    n_clusters = [3, 10]
                cluster = AutoCluster(criteria = modularity, init_k=n_clusters[0], max_k=n_clusters[1])
                labels = cluster.fit_predict(
                    X = aff, model = SpectralClustering, affinity='precomputed',
                    **kwargs
                )
            elif isinstance(n_clusters, int):
                cluster = SpectralClustering(n_clusters = n_clusters,  affinity='precomputed', **kwargs)
                labels = cluster.fit_predict(aff)
            comms = [np.where(labels == i)[0] for i in np.unique(labels)]
        elif clustering_method == 'networkx':
            try:
                import networkx as nx
            except ImportError:
                print('using networkx algorism need to install networkx >=3.0 package.')
                raise
            G = nx.from_numpy_array(aff)
            comms = nx.community.louvain.louvain_communities(G, **kwargs)
            comms = [np.array(list(i)) for i in comms if len(i) > 1]
        else:
            raise ValueError('clustering_method should be `sklearn` or `networkx`, or custom graph clustering genes after running `build_gene_graph_from_svgenes`.')
        labels = np.zeros(aff.shape[0],dtype=int)
        
        i = 0
        for comm in comms:
            if len(comm) <=3:
                labels[comm] = -1
            else:
                labels[comm] = i
                i += 1
        idx, counts = np.unique(labels,return_counts=True)
        if verbose:
            print(
                f'using {clustering_method} algorism get communities :',
                idx.size, '\n With size: ', counts, '\n'
            )
        if hasattr(self, 'result_df'):
            self.result_df['pattern'] = None
            self.result_df.loc[search_genes,'pattern'] = labels.astype(str)
        return labels


    def __possion_hypoth(self, X, verbose=False):
        def ASP_pdf(x, lam):
            from scipy.special import iv
            # Abosulte Substracted Possibility Distribution
            return (2 - (x == 0)) * np.exp(-2 * lam) * iv(x, 2 * lam)
        
        overall_max = X.max(axis=0)
        n_features = X.shape[1]
        lams = np.std(X, axis=0) ** 2
        overall_dist = [
            Distribution(
                value=np.arange(0, overall_max[i] + 1),
                prob=ASP_pdf(np.arange(0, overall_max[i] + 1), lams[i]),
            )
            for i in trange(n_features, desc="#1 Expected Dist within Possion Hypoth", disable=not verbose)
        ]

        return overall_dist, overall_max


    def __nodist_hypoth(self, X, verbose=False):
        n_features = X.shape[1]
        overall_dist = [
            Distribution().from_obs(obs=X[:, i]).dist_abs_subtract()
            for i in trange(n_features, desc="#1 Expected Dist within Nodist Hypoth", disable=not verbose)
        ]
        overall_max = X.max(axis=0)
        self.overall_dist = overall_dist
        self.overall_max = overall_max
        return overall_dist, overall_max


    def ent2gtest(self, Ents, ddof=0):
        n_obs = len(self.adata)
        # avoid the case of 0
        df = np.array([(np.array(list(d.values()))>1/n_obs).sum()-1 for d in self.overall_dist])
        pvals = chi2.sf(2 * n_obs * Ents, df - ddof)
        pvals[np.isnan(pvals)] = 1
        rejects, fdrs, _1, _2 = multipletests(pvals, method="fdr_bh")
        return {"pvals": pvals, "rejects": rejects, "fdrs": fdrs}


    def finding_spatial_neibors(self, K, finder=None):
        """Finding spatial neighbors.

        Args:
            K (int): Number of K neighbors.
            finder (str, optional): 'knn' or 'Delaunay'. Defaults to None.

        Returns:
            tuple: the indices of the neighbors.
        """
        if hasattr(self, "nbr_indices"):
            return self.nbr_indices

        finder = self.neighbor_finder if finder is None else finder

        if finder =='knn':
            nbr = NearestNeighbors(n_neighbors=K)
            nbr.fit(self.L)
            graph = nbr.kneighbors_graph()
            diag_mask = ~np.eye(*graph.shape).astype(bool)
            nbr_indices = np.where((graph == 1).todense() & diag_mask)
        elif finder =='Delaunay':
            tri = scipy.spatial.Delaunay(self.L)
            nbr_idx1 = np.zeros((0), dtype=int)
            nbr_idx2 = np.zeros((0), dtype=int)
            for i,j in combinations(range(tri.simplices.shape[1]),2):
                nbr_idx1 = np.append(nbr_idx1,tri.simplices[:,i])
                nbr_idx2 = np.append(nbr_idx2,tri.simplices[:,j])
            nbr_indices = (nbr_idx1,nbr_idx2)
            self.K = nbr_indices[0].size // self.L.shape[0] + 1

        self.nbr_indices = nbr_indices
        return nbr_indices
        
    
    def _AbsSubstract(self,X,indices,verbose):
        def computed_r(i):
            r = np.abs(X[indices[0], i] - X[indices[1], i])
            return np.unique(r, return_counts=True)

        Rs = Parallel(n_jobs=self.n_jobs)(
            delayed(computed_r)(i) for i in trange(X.shape[1], desc="#3 Computing Absolute Substract Value",disable=not verbose)
        )

        return Rs


    def select_top_K(self, num_select=None):
        """select top K genes based on the entropy.

        Args:
            num_select (int, optional): Number of genes want to select. Defaults to None.

        Raises:
            ValueError: `ent` is not defined, please run `fit` first.
        """
        if not hasattr(self, 'ent'):
            raise ValueError('`ent` is not defined, please run `fit` first.')
        if num_select is None:
            num_select = int(self.X.shape[1]*0.05) if num_select is None else num_select
            warnings.warn(f'num_select is not defined, will select top {num_select} genes.')
        if hasattr(self, 'rejects'):
            warnings.warn('`rejects` is already defined, will overwrite it.')
            # reject is a bool array
            threshold = np.sort(self.ent)[::-1][num_select]
            self.rejects = self.ent > threshold


    def fit(self, verbose=None, select_top_K = None):
        """Fitting the Spanve model.

        Args:
            verbose (optional): Defaults to None.
            select_top_K (int or None, optional): Number of genes want to select. Defaults to None, selecting the genes based on the FDR.

        Raises:
            ValueError: Unknown hypothesis type.

        Returns:
            Spanve: the self object.
        """
        # count time
        start = time.time()
        if verbose is None:
            verbose = self.verbose
        X = self.X
        n_features = X.shape[1]

        if self.hypoth_type == "possion":
            overall_dist, overall_max = self.__possion_hypoth(X, verbose=verbose)
        elif self.hypoth_type == "nodist":
            overall_dist, overall_max = self.__nodist_hypoth(X, verbose=verbose)
        else:
            raise ValueError("Unknown hypothesis type")

        # finding nearest k neighbors of each sample
        # from graph to get indices: recoder the index where the graph is 1
        indices = self.finding_spatial_neibors(K=self.K)

        if verbose:
            print("#2 Nearest Neighbors Found")

        Rs = self._AbsSubstract(X, indices, verbose)

        def computed_G(i):
            ind, counts = Rs[i]
            obs_dist = Distribution(value=ind, prob=counts / counts.sum())
            inds = np.arange(overall_max[i] + 1)
            x = obs_dist[inds]
            y = overall_dist[i][inds]
            ent = entropy(x, y)
            return ent

        Ents = np.array([computed_G(i) for i in range(n_features)])
        Ents[np.isnan(Ents)] = np.inf

        self.ent = Ents
        if verbose:
            print("#4 Entropy Calculated")

        gtest_result = self.ent2gtest(Ents)

        for k, v in gtest_result.items():
            setattr(self, k, v)
        
        if self.rejects.sum() < 1:
            warnings.warn(
            """
            WARNNING: no significant features found.
            -----------------------------------------------
            Adjusted `rejetcs` by top 5 persent genes (or defined select_top_K). You can still see fdrs in attribute `self.fdrs`.
            Or try to change params `neighbor_finder`; Or try to set a proper K (recommend to `int(0.1*n_cells/n_clusters)`). 
            Or number of observation are too small.
            """)
            self.select_top_K(num_select=select_top_K)
        
        if select_top_K is not None:
            self.select_top_K(select_top_K)
        
        if verbose:
            print("#5 G-test Performed")
        result_df = pd.DataFrame(
            {
                "ent": self.ent,
                "pvals": self.pvals,
                "rejects": self.rejects,
                "fdrs": self.fdrs,
                "max_expr": self.overall_max,
            },
            index= self.adata.var_names,
        )
        self.result_df = result_df
        self.adata.var['spanve_spatial_features'] = result_df['rejects']
        self.adata.uns['spanve_running_parmas'] = {
            key : getattr(self,key) for key in ['K','hypoth_type','neighbor_finder','n_jobs','verbose']
            }
        if verbose:
            print("Write results to adata.var['spanve_spatial_features']")
            print(f"#--- Done, using time {time.time()-start:.2f} sec ---#")
        
        return self
    
    
    def build_sample_graph_from_svgenes(self,alpha=0.05,select=None,K=None,verbose=None):
        """build sample graph from spatial genes.

        Args:
            alpha (float, optional): confidential level. Defaults to 0.05.
            select (array with bool dtype, optional): select genes, default to be `self.rejects`. Defaults to None.
            K (int, optional): Number of K neighbors, default to be `self.K`. Defaults to None.
            verbose (optional): Defaults to None.

        Returns:
            np.ndarray: the affinity matrix of the graph.
        """
        if verbose is None:
            verbose = self.verbose
        select = select if select is not None else self.rejects
        n_samples = self.X.shape[0]

        # --- neighborhood---
        if K is None or K == self.K:
            K = self.K
            nbr_indices = self.nbr_indices
        else:
            nbr_indices = self.finding_spatial_neibors(K)

        # --- computing ---

        graph = csr_array((n_samples,n_samples),dtype=int)
        for i in tqdm(np.where(select)[0],disable=not verbose,desc='generate graph from spatial genes'):
            thres = self.overall_dist[i].isf(n=self.overall_max[i],alpha=alpha)
            x = self.X[:,i]
            osx = np.abs(x[nbr_indices[0]] - x[nbr_indices[1]])
            idx = np.where(osx >= thres)[0]
            cell_id0 = nbr_indices[0][idx]
            cell_id1 = nbr_indices[1][idx]
            graph += csr_array((np.ones(len(cell_id0)),(cell_id0,cell_id1)),shape=(n_samples,n_samples),dtype=int)
        graph = graph + graph.T
        return graph


    def impute_from_graph(
        self,X,
        n_circle=2,
        graph=None,
        verbose=None,
        ):
        """Impute data from graph. By default, it will use the graph generated from `self.sample_graph_from_svgenes`.
        It is a graph convolution like process, with more impuation circles, the imputation will be more homologous. Not too much circles are recommended. It always to be 1 or 2.

        Args:
            X (np.ndarray): the input data.
            n_circle (int, optional): impute circles. Defaults to 2.
            graph (np.ndarray, optional): The graph affinity matrix. Defaults to None.
            verbose (bool, optional): Defaults to None.

        Returns:
            np.ndarray: The imputated data.
        """
        assert n_circle >= 0 and isinstance(n_circle,int), 'n_circle must be a positive integer'
        if n_circle == 0:
            return X
        X = self.X[:,self.rejects] if X is None else X
        n_samples = X.shape[0]
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print('Impute data, there are',n_circle,'circles')
        
        if graph is None:
            if hasattr(self, 'sample_graph_from_svgenes'):
                graph = self.sample_graph_from_svgenes
            else:
                graph = self.build_sample_graph_from_svgenes(alpha=0.05,select=self.rejects,K=self.K,verbose=verbose)
                self.sample_graph_from_svgenes = graph # saving computed graph for reuse.

        assert graph.shape == (n_samples,n_samples)

        graph = csr_array(np.eye(n_samples) + graph / (1+graph.sum(axis=0)))
        graph_imputed_X =( X.T @ graph ).T
        np.nan_to_num(graph_imputed_X,copy=False)
        
        if n_circle==1:
            return graph_imputed_X
        else:
            return self.impute_from_graph(X=graph_imputed_X,graph=graph,verbose=verbose,n_circle=n_circle-1)


    def plot_spatial(self,value,anndata=None,ax=None,):
        if anndata is None:
            anndata = self.adata
        if ax is None:
            fig,ax = plt.subplots()
        if value.dtype=='object':
            value = LabelEncoder().fit_transform(value)
        spatial_info = anndata.obsm['spatial']
        assert spatial_info.shape[0]==value.shape[0]
        ax.scatter(
            spatial_info[:,0],
            spatial_info[:,1],
            c=value,
            cmap='viridis',
            s=5,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # set color bar
        plt.colorbar(
            mappable= ax.collections[0],
            ax=ax)
        return ax


    def save(self,path,format='df'):
        """save the result to a file.

        Args:
            path (str): save path.
            format (str, optional): the saving format, 'df' or 'pickle'. Defaults to 'df'.

        Raises:
            ValueError: format must be `df` or `pickle`.
        """
        warnings.warn('This function will not save input data. If you want to save input data, please use `anndata.AnnData.write_h5ad` function.')
        if format == 'df':
            self.result_df.to_csv(path,index=True)
        elif format == 'pickle':
            attr = self.__dict__.copy()
            # delete adata and X
            del attr['adata']
            del attr['X']
            del attr['L']

            with open(path,'wb') as f:
                pickle.dump(attr,f)
        else:
            raise ValueError('format must be `df` or `pickle`')


    def load(self,obj,verbose=True):
        """load the result from a file.

        Args:
            obj (str, pd.DataFrame or dict obj): the file path or the object.
            verbose (bool, optional): Defaults to True.
        """
        if isinstance(obj, str):
            assert os.path.exists(obj), f'file {obj} not found'
            if obj.endswith('.pkl'):
                with open(obj,'rb') as f:
                    attr = pickle.load(f)
                self.__load_dict(attr,verbose=verbose)
            elif obj.endswith('.csv'):
                df = pd.read_csv(obj,index_col=0)
                self.__load_df(df, verbose)
        elif isinstance(obj, pd.DataFrame):
            self.__load_df(obj,verbose)
        elif isinstance(obj, dict):
            self.__load_dict(obj,verbose)


    def __load_df(self, df, verbose=True):
        self.result_df = df
        self.rejects = self.result_df['rejects'].values
        self.fdrs = self.result_df['fdrs'].values
        self.pvals = self.result_df['pvals'].values
        self.overall_max = self.result_df['max_expr'].values
        self.ent = self.result_df['ent'].values

        self.overall_dist,self.overall_max = self.__nodist_hypoth(self.X, verbose=verbose)
        self.nbr_indices = self.finding_spatial_neibors(self.K)


    def __load_dict(self,attr,verbose=True):

        def print_attr(name,a,b=None):
            if not verbose:
                return
            if type(a) in [int, str, float, bool]:
                if b is None:
                    print(f'Load {name}: {a}')
                else:
                    print(f'Load {name}: {a} -> {b}')
            elif type(a) in [list, tuple]:
                print(f'Load {name}: {type(a)} with length {len(a)}')
            
            elif type(a) in [np.ndarray,csr_array]:
                print(f'Load {name}: {type(a)} with shape {a.shape}')
            else:
                print(f'Load {name}: {type(a)}')

        # verbose of the changed attributes
        for k in attr.keys():
            if k in self.__dict__:
                print_attr(k,self.__dict__[k],attr[k])
                self.__dict__[k] = attr[k]
            else:
                self.__dict__[k] = attr[k]
                print_attr(k,attr[k])


class Spanve_gpu(Spanve):
    """Spanve model with GPU support. Note that the GPU is supported with `cupy` package.
    """
    def __init__(
        self,
        adata,
        K: int = None,
        device: int = 1,
        batch_size: int = 1024,
        hypoth_type: str = "nodist",
        neighbor_finder:str="knn", # or 'Delaunay'
        verbose=True):
        super().__init__(adata=adata,K=K,hypoth_type=hypoth_type,neighbor_finder=neighbor_finder,verbose=verbose)
        self.device = device
        self.batch_size = batch_size
    
    
    def _AbsSubstract(self,X,indices,verbose):
        try:
            import cupy as cp
            print(f'using cupy {cp.__version__} with {cp.cuda.Device(self.device).use()}')
        except:
            print(f'gpu is supportted by cupy package, follow the instruction (https://docs.cupy.dev/en/stable/install.html) to install cupy and set correct device id(now get {self.device}).')
            raise
        batch_size = self.batch_size
        n_features = X.shape[1]
        n_batches = int(np.ceil(n_features / batch_size))
        X = cp.array(X)
        indices = cp.array(indices)
        Rs = []


        def cpunique(x):
            y = cp.unique(x,return_counts=True)
            return y[0].get(),y[1].get()

        for i in tqdm(range(n_batches),disable=not verbose,desc=f'#3 Computing Absolute Substract Value(batch={batch_size})'):
            start = i * batch_size
            end = min((i+1) * batch_size,n_features)
            X_batch = X[:,start:end]
            Rs_batch = cp.abs(X_batch[indices[0],:] - X_batch[indices[1],:])
            Rs.extend([cpunique(Rs_batch[:,ii]) for ii in range(Rs_batch.shape[1])])
        return Rs
