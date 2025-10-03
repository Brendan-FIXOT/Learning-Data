import numpy as np
from typing import Optional, Dict, Any, Tuple

class LinearOutlierData:
    def __init__(self, n_samples = 100, outlier_ratio = 0.2, slope = 2.0,
                 intercept = 1.0, noise = 0.5, random_state = None):
        
        rng = np.random.default_rng(random_state)
        n_out = int(n_samples * outlier_ratio)
        n_in = n_samples - n_out

        # Inliers
        X_in = rng.uniform(0, 10, n_in)
        y_in = slope * X_in + intercept + rng.normal(0, noise, n_in)

        # Outliers
        X_out = rng.uniform(0, 10, n_out)
        y_out = rng.uniform(0, 25, n_out)

        # Concaténer et mélanger
        X = np.concatenate([X_in, X_out])
        y = np.concatenate([y_in, y_out])
        mask = np.concatenate([np.ones(n_in, dtype=bool), np.zeros(n_out, dtype=bool)])

        perm = rng.permutation(n_samples)
        self.X = X[perm]
        self.y = y[perm]
        self.inlier_mask = mask[perm]

        # Meta data
        self.meta: Dict[str, Any] = dict(
            slope=slope,
            intercept=intercept,
            n_inliers=n_in,
            n_outliers=n_out,
        )

    def as_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne X en (n,1) pour compatibilité avec sklearn."""
        return self.X.reshape(-1, 1), self.y
