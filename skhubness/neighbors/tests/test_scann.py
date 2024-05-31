# SPDX-License-Identifier: BSD-3-Clause

import pytest
import warnings

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils._testing import assert_raises
from sklearn.neighbors import NearestNeighbors

from skhubness.neighbors import ScaNNTransformer


def test_is_valid_estimator():
    check_estimator(ScaNNTransformer())


# @pytest.mark.xfail(reason="Annoy.Annoy can not be pickled as of v1.17")
def test_is_valid_estimator_in_main_memory():
    check_estimator(ScaNNTransformer(mmap_dir=None))
