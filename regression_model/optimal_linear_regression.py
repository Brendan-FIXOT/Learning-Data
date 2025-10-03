import numpy as np

class OptimalLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _as_2d_x(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit_linear_regression(self, X, y):
        X = self._as_2d_x(X)
        X_b = np.hstack([X, np.ones((X.shape[0], 1))]) # Add bias term
        a = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)     # y = a * X, to solve for a, a=(X^TX)^-1X^Ty
        self.coef_ = a[:-1]
        self.intercept_ = a[-1]
        return self

    def predict_linear_regression(self, X):
        X = self._as_2d_x(X)
        return X @ self.coef_ + self.intercept_ # y = a * X (for all samples)