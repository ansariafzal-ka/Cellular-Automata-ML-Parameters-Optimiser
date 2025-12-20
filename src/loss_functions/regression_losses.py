from .base_loss import LossFunction
import numpy as np

class MeanSquaredError(LossFunction):

    def compute_loss(self, y_true, y_predictions):
        return np.mean((y_true - y_predictions) ** 2)
    
    def compute_gradient(self, X, y_true, y_predictions):
        m = len(y_true)
        loss = y_predictions - y_true

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        dw = (2/m) * X.T.dot(loss)
        db = (2/m) * np.sum(loss)

        return np.concatenate([dw.flatten(), [db]])
    
class MeanAbsoluteError(LossFunction):

    def compute_loss(self, y_true, y_predictions):
        return np.mean(np.abs(y_true - y_predictions))
    
    def compute_gradient(self, X, y_true, y_predictions):
        m = len(y_true)
        loss = y_predictions - y_true
        signed_loss = np.sign(loss)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        dw = (1/m) * X.T.dot(signed_loss)
        db = (1/m) * np.sum(signed_loss)

        return np.concatenate([dw.flatten(), [db]])
    
class EpsilonInsensitiveLoss(LossFunction):

    def __init__(self, C, weights, epsilon=0.1):
        super().__init__()
        self.C = C
        self.weights = weights
        self.epsilon = epsilon

    def compute_loss(self, y_true, y_predictions):
        hinge_loss = np.sum(np.maximum(0, np.abs(y_true - y_predictions) - self.epsilon))
        margin = (np.linalg.norm(self.weights) ** 2) * 0.5

        return  margin + self.C * hinge_loss
    
    def compute_gradient(self, X, y_true, y_predictions):
        
        error = y_true - y_predictions
        mask = (np.abs(error) > self.epsilon).astype(float)

        dw_error = -self.C * X.T.dot(np.sign(error) * mask)
        dw = self.weights + dw_error

        db = -self.C * np.sum(np.sign(error) * mask)
    
        return np.concatenate([dw.flatten(), [db]])
    


        