import numpy as np

class LinearDataGenerator:
    def __init__(self, n_samples=1000, slope=2.0, intercept=1.0, noise=1.0, random_state=None):
        self.n_samples = n_samples
        self.slope = slope
        self.intercept = intercept
        self.noise = noise
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def generate(self, x_min=-10, x_max=10):
        X = np.linspace(x_min, x_max, self.n_samples)
        y = self.slope * X + self.intercept + self.rng.normal(0, self.noise, size=self.n_samples)
        return X, y

    def as_2d(self, x_min=-10, x_max=10):
        X, y = self.generate(x_min, x_max)
        return X.reshape(-1, 1), y
