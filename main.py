import numpy as np
import matplotlib.pyplot as plt
from regression_model.myransac import MyRANSAC
from regression_model.common_function import Common_methods
from regression_model.optimal_linear_regression import OptimalLinearRegression
from sklearn.linear_model import RANSACRegressor, LinearRegression
from data.data_provider import LinearOutlierData

def main():
    common_function = Common_methods()
    optimal_lr = OptimalLinearRegression()
    
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
    my_ransac.fit(X, y)
    model1 = my_ransac.model_
    inliers1 = my_ransac.inliers_
    
    # Run sklearn RANSAC
    print("Running sklearn RANSAC...")
    ransac = RANSACRegressor(estimator=LinearRegression(), max_trials=100, min_samples=2, residual_threshold=2.0, random_state=42).fit(X, y)
    inliers2 = ransac.inlier_mask_
    model2 = (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)  # slope, intercept
    
    # Run Linear Regression for reference
    lin_reg = optimal_lr.fit_linear_regression(X, y)
    ols_params = np.array([optimal_lr.coef_[0], optimal_lr.intercept_]) # slope, intercept
    
    # Plot results
    common_function.plot(X, y, model1=model1, inliers1=inliers1, model2=model2, model3=ols_params, inliers2=np.where(inliers2)[0], title="Comparison of MyRANSAC and Sklearn RANSAC")
    
    plt.show()

if __name__ == "__main__":
    main()