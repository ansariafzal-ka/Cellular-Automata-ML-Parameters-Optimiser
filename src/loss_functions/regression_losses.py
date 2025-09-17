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
    
class EpsilonIntensitiveLoss(LossFunction):

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def compute_loss(self, y_true, y_predictions):
        return np.mean(np.maximum(0, np.abs(y_true - y_predictions) - self.epsilon))
    
    def compute_gradient(self, X, y_true, y_predictions):
        m = len(y_true)
        loss = y_predictions - y_true
        abs_loss = np.abs(loss)
        mask = abs_loss > self.epsilon
        signed_loss = np.sign(loss) * mask

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        dw = (1/m) * X.T.dot(signed_loss)
        db = (1/m) * np.sum(signed_loss)
        return np.concatenate([dw.flatten(), [db]])
    


        