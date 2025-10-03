import numpy as np
import matplotlib.pyplot as plt
from myransac.myransac import MyRANSAC
from myransac.common_function import Common_methods
from sklearn.linear_model import RANSACRegressor, LinearRegression
from data.data_provider import LinearOutlierData

def main():
    common_function = Common_methods()
    
    # Retrieve synthetic dataset
    dataset = LinearOutlierData(
        n_samples=120,
        outlier_ratio=0.25,
        slope=2.0,
        intercept=1.0,
        noise=0.5,
        random_state=42,
    )
    
    X, y = dataset.as_2d()  # (n,1), compatible sklearn & ton RANSAC
    
    # Run our MyRANSAC implementation
    print("Running MyRANSAC implementation...")
    my_ransac = MyRANSAC(n_iters=100, threshold=2.0, min_sample=2)
    model1, inliers1 = my_ransac.fit(X, y)
    
    # Run sklearn RANSAC
    print("Running sklearn RANSAC...")
    ransac = RANSACRegressor(estimator=LinearRegression(), max_trials=100, min_samples=2, residual_threshold=2.0, random_state=42).fit(X, y)
    inliers2 = ransac.inlier_mask_
    model2 = (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)  # slope, intercept
    
    # Plot results
    common_function.plot(X, y, model1=model1, inliers1=inliers1, model2=model2, inliers2=np.where(inliers2)[0], title="Comparison of MyRANSAC and Sklearn RANSAC")
    
    plt.show()

if __name__ == "__main__":
    main()