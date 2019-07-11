[![PyPI](https://img.shields.io/pypi/v/hubness.svg)](
https://pypi.org/project/hubness)
[![Docs](https://readthedocs.org/projects/hubness/badge/?version=latest)](
https://hubness.readthedocs.io/en/latest/?badge=latest)
[![TravisCI](https://travis-ci.com/VarIr/hubness.svg?branch=master)](
https://travis-ci.com/VarIr/hubness)
[![Coverage](https://codecov.io/gh/VarIr/hubness/branch/master/graph/badge.svg?branch=master)](
https://codecov.io/gh/VarIr/hubness)
[![AppVeyorCI](https://ci.appveyor.com/api/projects/status/85bs46irwcwwbvyt/branch/master?svg=true)](
https://ci.appveyor.com/project/VarIr/hubness/branch/master)
[![Quality](https://img.shields.io/lgtm/grade/python/g/VarIr/hubness.svg?logo=lgtm&logoWidth=18)](
https://lgtm.com/projects/g/VarIr/hubness/context:python)
![License](https://img.shields.io/github/license/VarIr/hubness.svg)

# Hubness

(NOTE: THIS IS CURRENTLY UNDER HEAVY DEVELOPMENT. The API is not stable yet,
things might be broken here and there, docs are missing, etc.
A reasonably stable version is hopefully available soon,
and will then be uploaded to PyPI).

The `hubness` package comprises tools for the analysis and
reduction of hubness in high-dimensional data.
Hubness is an aspect of the _curse of dimensionality_
and is detrimental to many machine learning and data mining tasks.

The `hubness.analysis` and `hubness.reduction` package allows you to

- analyze, whether your data sets show hubness
- reduce hubness via a variety of different techniques 
- perform evaluation tasks with both internal and external measures

The `hubness.neighbors` package acts as a drop-in replacement for `sklearn.neighbors`.
In addition to the functionality inherited from `scikit-learn`,
it also features
- _approximate nearest neighbor_ search
- hubness reduction
- and combinations,

which allows for fast hubness reduced neighbor search in large datasets
(tested with >1M objects).

We try to follow the API conventions and code style of scikit-learn.

## Installation


Make sure you have a working Python3 environment (at least 3.7).

Use pip3 to install the latest stable version of `hubness` from PyPI:

```bash
pip3 install hubness
```

`hubness` requires `numpy`, `scipy` and `scikit-learn` packages.
Approximate nearest neighbor search and approximate hubness reduction
additionally requires `nmslib` and/or `falconn`.
Some modules require `pandas` or `joblib`. All these packages are available
from open repositories, such as PyPI, and are installed automatically, if necessary.

For more details and alternatives, please see the [Installation instructions](
http://hubness.readthedocs.io/en/latest/user/installation.html).

## Documentation

Documentation is available online: 
http://hubness.readthedocs.io/en/latest/index.html

## Quickstart

Users of `hubness` may want to 

1. analyse, whether their data show hubness
2. reduce hubness
3. perform learning (classification, regression, ...)

The following example shows all these steps for an example dataset
from the text domain (dexter). (Please make sure you have installed `hubness`).

```python
# load the example dataset 'dexter'
from hubness.data import load_dexter
X, y = load_dexter()

# dexter is embedded in a high-dimensional space,
# and could, thus, be prone to hubness
print(f'X.shape = {X.shape}, y.shape={y.shape}')

# assess the actual degree of hubness in dexter
from hubness import Hubness
hub = Hubness(k=5, metric='cosine')
hub.fit_transform(X)
print(f'Skewness = {hub.k_skewness_:.3f}')

# additional hubness indices are available, for example:
print(f'Robin hood index: {hub.robinhood_index_:.3f}')
print(f'Antihub occurrence: {hub.antihub_occurrence_:.3f}')
print(f'Hub occurrence: {hub.hub_occurrence_:.3f}')

# There is considerable hubness in dexter.
# Let's see, whether hubness reduction can improve
# kNN classification performance 
from sklearn.model_selection import cross_val_score
from hubness.neighbors import KNeighborsClassifier

# vanilla kNN
knn_standard = KNeighborsClassifier(n_neighbors=5,
                                    metric='cosine')
acc_standard = cross_val_score(knn_standard, X, y, cv=5)

# kNN with hubness reduction (mutual proximity)
knn_mp = KNeighborsClassifier(n_neighbors=5,
                              metric='cosine',
                              hubness='mutual_proximity')
acc_mp = cross_val_score(knn_mp, X, y, cv=5)

print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')
print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')

# Accuracy was considerably improved by mutual proximity.
# Did it actually reduce hubness?
knn_mp.fit(X, y)
neighbor_graph = knn_mp.kneighbors_graph()

hub_mp = Hubness(k=5, metric='precomputed').estimate(neighbor_graph)
print(f'Skewness: {hub_mp.k_skewness_:.3f} '
      f'(reduction of {hub.k_skewness_ - hub_mp.k_skewness_:.3f})')
print(f'Robin hood: {hub_mp.robinhood_index_:.3f} '
      f'(reduction of {hub.robinhood_index_ - hub_mp.robinhood_index_:.3f})')

# The neighbor graph can also be created directly,
# with or without hubness reduction
from hubness.neighbors import kneighbors_graph
neighbor_graph = kneighbors_graph(X, n_neighbors=5, hubness='mutual_proximity')
```

Check the [Tutorial](http://hubness.readthedocs.io/en/latest/user/tutorial.html)
for in-depth explanations of the same. 


## Development

The `hubness` package is a work in progress. Get in touch with us if you have
comments, would like to see an additional feature implemented, would like
to contribute code or have any other kind of issue. Please don't hesitate
to file an [issue](https://github.com/VarIr/hubness/issues)
here on GitHub. 

    (c) 2018-2019, Roman Feldbauer
    Austrian Research Institute for Artificial Intelligence (OFAI) and
    University of Vienna, Division of Computational Systems Biology (CUBE)
    Contact: <roman.feldbauer@univie.ac.at>

## Citation

A software publication paper is currently in preparation. Until then,
if you use the `hubness` package in your scientific publication, please cite:

    @INPROCEEDINGS{8588814,
    author={R. {Feldbauer} and M. {Leodolter} and C. {Plant} and A. {Flexer}},
    booktitle={2018 IEEE International Conference on Big Knowledge (ICBK)},
    title={Fast Approximate Hubness Reduction for Large High-Dimensional Data},
    year={2018},
    volume={},
    number={},
    pages={358-367},
    keywords={computational complexity;data analysis;data mining;mobile computing;public domain software;software packages;mobile device;open source software package;high-dimensional data mining;fast approximate hubness reduction;massive mobility data;linear complexity;quadratic algorithmic complexity;dimensionality curse;Complexity theory;Indexes;Estimation;Data mining;Approximation algorithms;Time measurement;curse of dimensionality;high-dimensional data mining;hubness;linear complexity;interpretability;smartphones;transport mode detection},
    doi={10.1109/ICBK.2018.00055},
    ISSN={},
    month={Nov},}

The technical report `Fast approximate hubness reduction for large high-dimensional data`
is available at [OFAI](http://www.ofai.at/cgi-bin/tr-online?number+2018-02).

Additional reading

`Local and Global Scaling Reduce Hubs in Space`, Journal of Machine Learning Research 2012,
[Link](http://www.jmlr.org/papers/v13/schnitzer12a.html).

`A comprehensive empirical comparison of hubness reduction in high-dimensional spaces`,
Knowledge and Information Systems 2018, [DOI](https://doi.org/10.1007/s10115-018-1205-y).

License
-------
The `hubness` package is licensed under the terms of the [GNU GPLv3](LICENSE.txt).

The `hubness.neighbors` package was modified from `sklearn.neighbors`,
licensed under the terms of BSD-3 (see [LICENSE](external/SCIKIT_LEARN_LICENSE.txt)).

Acknowledgements
----------------
Several parts of `hubness` adapt code from `scikit-learn`. We thank all the authors
and contributors of this project for the tremendous work they have done.

PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com
