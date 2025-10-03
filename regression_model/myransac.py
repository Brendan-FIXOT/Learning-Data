import numpy as np
import matplotlib.pyplot as plt
import random
from .common_function import Common_methods
from .optimal_linear_regression import OptimalLinearRegression

class MyRANSAC(Common_methods, OptimalLinearRegression):
    def __init__(self, n_iters=100, threshold=None, min_sample=2, random_state=None):
        self.model_ = None
        self.inliers_ = None
        self.n_iters = n_iters
        self.min_sample = min_sample
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.optimal_lr = OptimalLinearRegression()
        
        # Si threshold = None → calcul adaptatif basé sur MAD
        if threshold is None:
            median = np.median(y)
            mad = np.median(np.abs(y - median))
            self.threshold = 1.4826 * mad
            print(f"[INFO] Adaptive threshold set to {threshold:.3f}")
        else:
            self.threshold = threshold
    
    def fit(self, X, y):
        best_inliers = []
        best_model = None
        
        for i in range(self.n_iters):
            # Initialize 2 random points
            sample = self.rng.choice(len(X), size=self.min_sample, replace=False)
            self.optimal_lr.fit_linear_regression(X[sample], y[sample])

            y_pred = self.optimal_lr.predict_linear_regression(X)

            error = np.abs(y - y_pred)
               
            inliers = np.where(error < self.threshold)[0] # Because np.where() give a tuple
            
            if (len(inliers) > len(best_inliers)):
                best_inliers = inliers # New best inliers
                best_model = self.optimal_lr.fit_linear_regression(X[best_inliers], y[best_inliers]) # New best model trained
                best_model = np.array([best_model.coef_[0], best_model.intercept_])

        self.model_ = best_model
        self.inliers_ = best_inliers
        return self
