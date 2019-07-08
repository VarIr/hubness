from abc import ABC, abstractmethod
import warnings


class ApproximateNearestNeighbor(ABC):
    """ Abstract base class for approximate nearest neighbor search methods. """

    def __init__(self, n_candidates: int = 5, metric: str = 'sqeuclidean',
                 n_jobs: int = 1, verbose: int = 0, *args, **kwargs):
        self.n_candidates = n_candidates
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def kneighbors(self, X, n_candidates, return_distance):
        pass


class UnavailableANN(ApproximateNearestNeighbor, ABC):
    """ Placeholder for ANN methods that are not available on specific platforms. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warning.warn(f'The chosen approximate nearest neighbor method is not supported on your platform.')
