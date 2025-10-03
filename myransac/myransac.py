import numpy as np
import matplotlib.pyplot as plt
import random
from .common_function import Common_methods

class MyRANSAC(Common_methods):
    def __init__(self, n_iters=100, threshold=None, min_sample=2):
        self.model_ = None
        self.inliers_ = None
        self.n_iters = n_iters
        self.min_sample = min_sample
        
        # Si threshold = None → calcul adaptatif basé sur MAD
        if threshold is None:
            median = np.median(y)
            mad = np.median(np.abs(y - median))
            self.threshold = 1.4826 * mad
            print(f"[INFO] Adaptive threshold set to {threshold:.3f}")
        else:
            self.threshold = threshold
        
    def fit_linear_regression(self, X, y):
        if X.ndim == 1:   # if X is 1D → reshape to column
            X = X.reshape(-1, 1)
        X_b = np.hstack([X, np.ones((X.shape[0], 1))]) # Add bias term
        a = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y # y = a * X, to solve for a, a=(X^TX)^-1X^Ty 
        return a

    def predict_linear_regression(self, X, a):
        if X.ndim == 1:  
            X = X.reshape(-1, 1)
        X_b = np.hstack([X, np.ones((X.shape[0], 1))]) # add bias term
        return X_b @ a # y = a * X (for all samples)
    
    def fit(self, X, y):
        best_inliers = []
        best_model = None 
        
        for i in range(self.n_iters):
            # Initialize 2 random points
            sample = random.sample(range(len(X)), self.min_sample) # or np.random.default_rng() if you want a seed to reproduce results

            model = self.fit_linear_regression(X[sample], y[sample])

            y_pred = self.predict_linear_regression(X, model)

            error = np.abs(y - y_pred)
               
            inliers = np.where(error < self.threshold)[0] # Because np.where() give a tuple
            
            if (len(inliers) > len(best_inliers)):
                best_inliers = inliers # New best inliers
                best_model = self.fit_linear_regression(X[best_inliers], y[best_inliers]) # New best model trained
            
        return best_model, best_inliers