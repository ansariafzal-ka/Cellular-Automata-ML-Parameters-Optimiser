from .base_model import Model
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(Model):

    def __init__(self, n_features):
        super().__init__()
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.bounds = [(-1000.0, 1000.0)] * self.get_param_count() # if there are 3 params then the result will be [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

    def predict(self, X):
        return X @ self.weights + self.bias
    
    def get_params(self):
        return np.concatenate([self.weights, [self.bias]])
    
    def set_params(self, params):
        self.weights = params[:-1]
        self.bias = params[-1]

    def set_param_bounds(self, bounds):
        self.bounds = bounds * self.get_param_count()

    def get_param_count(self):
        return len(self.weights) + 1

    def get_param_bounds(self):
        return self.bounds


class LogisticRegression(Model):

    def __init__(self, n_features, threshold=0.5):
        super().__init__()
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.threshold = threshold
        self.bounds = [(-5.0, 5.0)] * self.get_param_count()

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_labels(self, X):
        linear_model = X @ self.weights + self.bias
        y_predicted_proba = self._sigmoid(linear_model)
        y_predicted_classes = (y_predicted_proba > self.threshold).astype(int)

        return y_predicted_classes
    
    def predict(self, X):
        linear_model = X @ self.weights + self.bias
        y_predicted_proba = self._sigmoid(linear_model)

        return y_predicted_proba

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
    
class SoftmaxRegression(Model):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.n_classes = n_classes
        self.n_features = n_features
        self.bounds = [(-10.0, 10.0)] * self.get_param_count()

    def _softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, X):
        linear_model = X @ self.weights + self.bias
        y_predicted_proba = self._softmax(linear_model)

        return y_predicted_proba
    
    def predict_labels(self, X):
        linear_model = X @ self.weights + self.bias
        y_predicted_proba = self._softmax(linear_model)
        y_predicted_classes = np.argmax(y_predicted_proba, axis=1)

        return y_predicted_classes
    
    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.bias.flatten()])
    
    def set_params(self, params):
        n_weight_params = self.n_features * self.n_classes
        
        self.weights = params[:n_weight_params].reshape(self.n_features, self.n_classes)
        self.bias = params[n_weight_params:] 

    def get_param_count(self):
        return self.n_features * self.n_classes + self.n_classes
    
    def set_param_bounds(self, bounds):
        self.bounds = bounds * self.get_param_count()
    
    def get_param_bounds(self):
        return self.bounds