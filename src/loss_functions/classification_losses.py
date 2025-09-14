from .base_loss import LossFunction
import numpy as np

class BinaryCrossEntropy(LossFunction):
    def compute_loss(self, y_true, y_prediction) :
        y_prediction = np.clip(y_prediction, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_prediction) + (1 - y_true) * np.log(1 - y_prediction))
    
    def compute_gradient(self, X, y_true, y_prediction):
        m = len(y_true)
        loss = y_prediction - y_true
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        dw = (1/m) * X.T.dot(loss)
        db = (1/m) * np.sum(loss)
        return np.concatenate([dw.flatten(), [db]])
    
class CategoricalCrossEntropy(LossFunction):
    def compute_loss(self, y_true, y_prediction):
        y_prediction = np.clip(y_prediction, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_prediction), axis=1))
    
    def compute_gradient(self, X, y_true, y_prediction):
        m = len(y_true)
        loss = y_prediction - y_true
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        dw = (1/m) * X.T.dot(loss)
        db = (1/m) * np.sum(loss, axis=0)
        return np.concatenate([dw.flatten(), db.flatten()])
    
class HingeLoss(LossFunction):
    def compute_loss(self, y_true, y_predictions):
        y_true = np.where(y_true == 0, -1, 1)
        return np.mean(np.maximum(0, 1 - y_true * y_predictions))
    
    def compute_gradient(self, X, y_true, y_prediction):
        m = len(y_true)
        y_true = np.where(y_true == 0, -1, 1)
        margin = y_true * y_prediction
        mask = margin < 1
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        dw = -(1/m) * X.T.dot(y_true * mask)
        db = -(1/m) * np.sum(y_true * mask)
        return np.concatenate([dw.flatten(), [db]])
        