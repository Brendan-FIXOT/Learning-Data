import matplotlib.pyplot as plt

class Common_methods:
    def __init__(self):
        self.model_ = None
        
    def plot(self, X, y, model1=None, model2=None, model3=None, inliers1=None, inliers2=None, ax=None, title=None, show=True):
        import numpy as np
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()

        created_ax = False
        if ax is None:
            fig, ax = plt.subplots()
            created_ax = True

        # Points inliers1 / outliers2
        if inliers1 is not None and len(inliers1) > 0:
            mask = np.zeros_like(X, dtype=bool)
            mask[inliers1] = True
            ax.scatter(X[~mask], y[~mask], s=20, marker="+", label="outliers1", color="C3", alpha=0.6)
            ax.scatter(X[mask], y[mask], s=20, marker="+", label="inliers1", color="C2")
        else:
            ax.scatter(X, y, s=20, marker="+", label="data")
            
        # Points inliers2 / outliers2
        if inliers2 is not None and len(inliers2) > 0:
            mask2 = np.zeros_like(X, dtype=bool)
            mask2[inliers2] = True
            ax.scatter(X[~mask2], y[~mask2], s=10, label="outliers2", color="C7", alpha=0.6)
            ax.scatter(X[mask2], y[mask2], s=10, label="inliers2", color="C6")
        else:
            ax.scatter(X, y, s=10, label="data")

        xx = np.linspace(X.min(), X.max(), 100)

        # Premier modèle
        if model1 is not None:
            m1 = np.asarray(model1).ravel()
            if m1.size >= 2:
                yy = m1[0] * xx + m1[-1]
                ax.plot(xx, yy, color="k", lw=2, linestyle="--", label="model1")

        # Deuxième modèle
        if model2 is not None:
            m2 = np.asarray(model2).ravel()
            if m2.size >= 2:
                yy2 = m2[0] * xx + m2[-1]
                ax.plot(xx, yy2, color="C1", lw=2, linestyle="--", label="model2")
                
        # Troisième modèle (OLS)
        if model3 is not None:
            m3 = np.asarray(model3).ravel()
            if m3.size >= 2:
                yy3 = m3[0] * xx + m3[-1]
                ax.plot(xx, yy3, color="k", lw=2, linestyle=":", label="OLS")

        if title:
            ax.set_title(title)
        ax.legend()

        if show and created_ax:
            plt.show()
        return ax
