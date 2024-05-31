# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import logging

try:
    import scann
except ImportError:
    scann = None  # pragma: no cover

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..utils.io import create_tempfile_preferably_in_dir

__all__ = [
    "ScaNNTransformer",
]


class ScaNNTransformer(BaseEstimator, TransformerMixin):
    """ Approximate nearest neighbors retrieval with ScaNN.

    Compatible with sklearn's KNeighborsTransformer.
    ScaNN [1]_ is an approximate nearest neighbor library by Google Research,
    which "includes search space pruning and quantization for Maximum Inner Product Search
    and also supports other distance functions such as Euclidean distance."

    Parameters
    ----------
    n_neighbors: int
        Number of neighbors to retrieve
    metric: str
        Distance metric, allowed are "dot_product", "euclidean", TODO.
    ...
    mmap_dir: str
        Memory-map the index to the given directory. This may be required to make the class pickleable.
        If None, keep everything in main memory (NON pickleable index),
        if mmap_dir is a string, it is interpreted as a directory to store the index into,
        if "auto", create a temp dir for the index, preferably in /dev/shm on Linux.
    verbose: int
        If verbose > 0, show a progress bar on fit/indexing and transform/querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures

    References
    ----------
    .. [1] Guo et al., (2020).
           "Accelerating Large-Scale Inference with Anisotropic Vector Quantization."
           International Conference on Machine Learning (ICML).
           http://proceedings.mlr.press/v119/guo20h/guo20h.pdf
    """
    _pkg_name = "scann"
    valid_metrics = ["dot_product", "squared_l2", ]  # TODO

    def __init__(
            self,
            n_neighbors: int = 5,
            metric: str = "dot_product",
            *,
            dimensions_per_block: int = 2,
            reordering_num_neighbors: int = 100,
            leaves_to_search: int = 150,
            num_leaves: int = 2_000,
            num_leaves_to_search: int = 100,
            min_partition_size=50,
            training_sample_size: int = 250_000,
            anisotropic_quantization_threshold: float = float("nan"),
            min_cluster_size: int = 100,
            hash_type: str = "lut16",
            training_iterations: int = 10,
            final_num_neighbors: int = None,
            pre_reorder_num_neighbors: int = None,
            batch_size: int = 256,
            mmap_dir: str = None,
            parallel: bool = False,
            verbose: int = 0,
    ):
        if scann is None:  # pragma: no cover
            raise ImportError(
                f"Please install the {type(self)._pkg_name} package before using {type(self).__name__}:\n"
                f"pip install {type(self)._pkg_name}",
            ) from None

        self.n_neighbors = n_neighbors
        self.metric = metric or "dot_product"
        self.reordering_num_neighbors = reordering_num_neighbors
        # Tree parameters
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.min_partition_size = min_partition_size
        self.training_sample_size = training_sample_size
        # score_ah parameters
        self.dimensions_per_block = dimensions_per_block
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold
        self.min_cluster_size = min_cluster_size
        self.hash_type = hash_type
        self.training_iterations = training_iterations
        # Search kwargs
        self.final_num_neighbors = final_num_neighbors
        self.pre_reorder_num_neighbors = pre_reorder_num_neighbors
        self.leaves_to_search = leaves_to_search
        self.batch_size = batch_size

        self.parallel = parallel
        self.mmap_dir = mmap_dir
        self.verbose = verbose

    def fit(self, X, y=None) -> ScaNNTransformer:
        """ Build the ScaNN.Index and insert data from X.  TODO

        Parameters
        ----------
        X: array-like
            Data to be indexed
        y: ignored

        Returns
        -------
        self: ScaNNTransformer
            An instance of ScaNNTransformer with a built index
        """
        X: np.ndarray = check_array(X)
        n_samples, n_features = X.shape
        self.dtype_ = X.dtype
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features
        n_neighbors = self.n_neighbors + 1

        space = {
            **{x: x for x in type(self).valid_metrics},
            None: "dot_product",  # default
            "euclidean": "squared_l2",
            "sqeuclidean": "squared_l2",
            "angular": "dot_product",
        }.get(self.metric, None)
        if space is None:
            raise ValueError(f"Invalid metric: {self.metric}")
        self.space_ = space
        self._do_sqrt = self.metric == "sqeuclidean"

        if isinstance(self.mmap_dir, str):
            directory = "/dev/shm" if self.mmap_dir == "auto" else self.mmap_dir
            self.neighbor_index_ = create_tempfile_preferably_in_dir(
                prefix="skhubness_",
                suffix=".scann",
                directory=directory,
            )
            if self.mmap_dir == "auto":
                logging.warning(
                    f"The index will be stored in {self.neighbor_index_}. "
                    f"It will NOT be deleted automatically, when this instance is destructed.",
                )
        else:  # e.g. None
            self.mmap_dir = None

        norm = np.linalg.norm(X, axis=1)
        norm[norm == 0] = 1  # avoid division by zero
        normalized_dataset = X / norm[:, np.newaxis]
        # configure ScaNN as a tree - asymmetric hash hybrid with reordering
        # anisotropic quantization as described in the paper; see README

        # Adjust parameters to the number of samples, if necessary
        num_leaves = min(self.num_leaves, n_samples)
        num_leaves_to_search = min(self.num_leaves_to_search, n_samples)
        min_cluster_size = min(self.min_cluster_size, n_samples)
        training_sample_size = min(self.training_sample_size, n_samples)
        min_partition_size = min(self.min_partition_size, n_samples)

        # TODO make adjustible (brute force for small n; partitioning for large n; AH etc.)
        # TODO also, fix estimator checks for very small n
        searcher = scann.scann_ops_pybind.builder(
            normalized_dataset,
            num_neighbors=n_neighbors,
            distance_measure=self.space_,
        ).tree(
            num_leaves,
            num_leaves_to_search,
            training_sample_size=training_sample_size,
            min_partition_size=min_partition_size,
            training_iterations=self.training_iterations,
        ).score_ah(
            self.dimensions_per_block,
            anisotropic_quantization_threshold=self.anisotropic_quantization_threshold,
            training_sample_size=training_sample_size,
            min_cluster_size=min_cluster_size,
            hash_type=self.hash_type,
            training_iterations=self.training_iterations,
        ).reorder(
            self.reordering_num_neighbors,
        ).build()

        if self.mmap_dir is None:
            self.neighbor_index_ = searcher
        else:
            raise NotImplementedError("Memory-mapping is not yet implemented for ScaNN.")

        return self

    def transform(self, X) -> csr_matrix:
        """ Create k-neighbors graph for the query objects in X.

        Parameters
        ----------
        X : array-like
            Query objects

        Returns
        -------
        kneighbors_graph : csr_matrix
            The retrieved approximate nearest neighbors in the index for each query.
        """
        check_is_fitted(self, "neighbor_index_")
        # if X is None:
        #     n_samples_transform = self.n_samples_in_
        # else:
        X: np.ndarray = check_array(X)
        if X.dtype == np.float64:
            X = X.astype(np.float64)
        n_samples_transform, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(f"Shape of X ({n_features} features) does not match "
                             f"shape of fitted data ({self.n_features_in_} features.")
        n_neighbors = self.n_neighbors + 1

        search_kwargs = {
            "final_num_neighbors": self.final_num_neighbors,
            "pre_reorder_num_neighbors": self.pre_reorder_num_neighbors,
            "leaves_to_search": self.leaves_to_search,
        }
        if self.parallel:
            search_func = self.neighbor_index_.search_batched_parallel
            search_kwargs["batch_size"] = self.batch_size
        else:
            search_func = self.neighbor_index_.search_batched
        neighbors, distances = search_func(X, **search_kwargs)

        # ScaNN does squared L2 only. If user requested Euclidean, we need to take the sqrt.
        if self._do_sqrt:
            np.sqrt(distances, out=distances)

        indptr = np.arange(
            start=0,
            stop=n_samples_transform * n_neighbors + 1,
            step=n_neighbors,
        )
        kneighbors_graph = csr_matrix(
            (distances.ravel(), neighbors.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_in_),
        )

        return kneighbors_graph

    def fit_transform(self, X, y=None, **fit_params):
        """ Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Training data
        y : ignored
        fit_params : dict
            Additional fit parameters

        Returns
        -------
        kneighbors_graph : csr_matrix
            The retrieved approximate nearest neighbors in the index for each query.
        """
        return self.fit(X).transform(X)
