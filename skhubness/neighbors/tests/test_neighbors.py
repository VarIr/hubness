import sys

import pytest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsTransformer, KNeighborsClassifier

from skhubness.neighbors import AnnoyTransformer
from skhubness.neighbors import NMSlibTransformer
from skhubness.neighbors import NGTTransformer
from skhubness.neighbors import PuffinnTransformer
from skhubness.neighbors import ScaNNTransformer

NON_WINDOWS = (NGTTransformer, PuffinnTransformer, ScaNNTransformer)


@pytest.mark.parametrize("n_neighbors", [1, 5, 10])
@pytest.mark.parametrize("metric", [None, "euclidean", "cosine"])
@pytest.mark.parametrize("ApproximateNNTransformer",
                         [AnnoyTransformer, NGTTransformer, NMSlibTransformer, PuffinnTransformer, ScaNNTransformer])
def test_ann_transformers_similar_to_exact_transformer(ApproximateNNTransformer, n_neighbors, metric):
    if sys.platform == "win32" and issubclass(ApproximateNNTransformer, NON_WINDOWS):
        pytest.skip(f"{ApproximateNNTransformer.__name__} is not available on Windows.")
    knn_metric = metric
    ann_metric = metric
    if issubclass(ApproximateNNTransformer, PuffinnTransformer) and metric in ["euclidean", "cosine"]:
        pytest.skip(f"{ApproximateNNTransformer.__name__} does not support metric={metric}")
    if issubclass(ApproximateNNTransformer, AnnoyTransformer) and metric == "cosine":
        ann_metric = "angular"
    if issubclass(ApproximateNNTransformer, ScaNNTransformer) and metric == "cosine":
        ann_metric = "dot_product"
    n_samples = 100
    X, y = make_classification(
        n_samples=n_samples,
        random_state=123,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=456, shuffle=True, stratify=y)

    # Exackt kNN graph for comparison
    kwargs = {}
    if knn_metric is not None:
        kwargs["metric"] = knn_metric
    knn = KNeighborsTransformer(n_neighbors=n_neighbors, **kwargs)
    graph_train = knn.fit_transform(X_train, y_train)
    knn_graph: csr_matrix = knn.transform(X_test)
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed")
    y_pred_knn = knn_clf.fit(graph_train, y_train).predict(knn_graph)
    knn_acc = accuracy_score(y_true=y_test, y_pred=y_pred_knn)

    # ANN graph
    kwargs = {}
    if ann_metric is not None:
        kwargs["metric"] = ann_metric
    ann = ApproximateNNTransformer(n_neighbors=n_neighbors, **kwargs)
    graph_train = ann.fit_transform(X_train, y_train)
    ann_graph = ann.transform(X_test)
    ann_clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed")
    y_pred_ann = ann_clf.fit(graph_train, y_train).predict(ann_graph)
    ann_acc = accuracy_score(y_true=y_test, y_pred=y_pred_ann)

    # Neighbor graphs should be same class, same shape, same dtype
    assert ann_graph.__class__ == knn_graph.__class__
    assert ann_graph.shape == knn_graph.shape
    assert ann_graph.dtype == knn_graph.dtype or ApproximateNNTransformer is ScaNNTransformer
    assert ann_graph.nnz == knn_graph.nnz
    if issubclass(ApproximateNNTransformer, (AnnoyTransformer, ScaNNTransformer)):
        pass  # Known inaccuracy
    elif issubclass(ApproximateNNTransformer, PuffinnTransformer) and metric is None:
        pass  # Known inaccuracy
    else:
        np.testing.assert_array_equal(ann_graph.indices.ravel(), knn_graph.indices.ravel())
        np.testing.assert_array_almost_equal(ann_graph.data.ravel(), knn_graph.data.ravel())
    if issubclass(ApproximateNNTransformer, AnnoyTransformer) and metric == "cosine" and n_neighbors == 1:
        return  # Known inaccurate result
    assert ann_acc > knn_acc or np.isclose(ann_acc, knn_acc), "ApproximateNN accuracy << exact kNN accuracy."
