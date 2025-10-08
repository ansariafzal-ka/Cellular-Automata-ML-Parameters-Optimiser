from .base_model import Model
import numpy as np


class SupportVectorClassifier(Model):
    def __init__(self, n_features):
        super().__init__()
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def predict(self, X):
        raw_predictions = X.dot(self.weights) + self.bias
        return np.where(raw_predictions >= 0, 1, 0)

    def get_params(self):
        return np.concatenate([self.weights, [self.bias]])

    def set_params(self, params):
        self.weights = params[:-1]
        self.bias = params[-1]

    def get_param_count(self):
        return len(self.weights) + 1

    def get_param_bounds(self):
        num_params = self.get_param_count()
        return [(-100.0, 100.0)] * num_params
    
class SupportVectorRegression(Model):
    def __init__(self, n_features):
        super().__init__()
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def predict(self, X):
        return X.dot(self.weights) + self.bias

    def get_params(self):
        return np.concatenate([self.weights, [self.bias]])

    def set_params(self, params):
        self.weights = params[:-1]
        self.bias = params[-1]

    def get_param_count(self):
        return len(self.weights) + 1

    def get_param_bounds(self):
        num_params = self.get_param_count()
        return [(-100.0, 100.0)] * num_params