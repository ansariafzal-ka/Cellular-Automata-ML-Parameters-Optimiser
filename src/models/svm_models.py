from .base_model import Model
from ..loss_functions.classification_losses import HingeLoss
import numpy as np


class SupportVectorClassifier(Model):
    def __init__(self, n_features, C=1.0):
        super().__init__()
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.C = C
        self.bounds = [(-1000.0, 1000.0)] * self.get_param_count()
        self.loss_fn = HingeLoss(self.C, self.weights)

    def predict(self, X):
        raw_predictions = X.dot(self.weights) + self.bias
        return np.where(raw_predictions >= 0, 1, -1)

    def get_params(self):
        return np.concatenate([self.weights, [self.bias]])

    def set_params(self, params):
        self.weights = params[:-1]
        self.bias = params[-1]

    def get_param_count(self):
        return len(self.weights) + 1
    
    def set_param_bounds(self, bounds):
        self.bounds = bounds * self.get_param_count()

    def get_param_bounds(self):
        return  self.bounds
    
class SupportVectorRegression(Model):
    def __init__(self, n_features):
        super().__init__()
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.bounds = [(-1000.0, 1000.0)] * self.get_param_count()


    def predict(self, X):
        return X.dot(self.weights) + self.bias

    def get_params(self):
        return np.concatenate([self.weights, [self.bias]])

    def set_params(self, params):
        self.weights = params[:-1]
        self.bias = params[-1]

    def get_param_count(self):
        return len(self.weights) + 1
    
    def set_param_bounds(self, bounds):
        self.bounds = bounds * self.get_param_count()

    def get_param_bounds(self):
        return self.bounds