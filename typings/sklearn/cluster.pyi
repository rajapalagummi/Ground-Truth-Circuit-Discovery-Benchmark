from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

class KMeans:
    cluster_centers_: NDArray[np.float64]
    labels_: NDArray[np.int64]
    def __init__(self, n_clusters: int = ..., *, n_init: int | str = ..., random_state: int | None = ...) -> None: ...
    def fit(self, X: NDArray[Any]) -> Self: ...
