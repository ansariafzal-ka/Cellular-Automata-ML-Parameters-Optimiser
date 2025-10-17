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

    def _to_one_hot(self, y, n_classes):
        """Internal helper to convert integer labels to one-hot."""
        # Use a property of y_prediction to get the number of classes
        y_one_hot = np.zeros((len(y), n_classes))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

    def compute_loss(self, y_true, y_prediction):
        if y_true.ndim == 1:
            n_classes = y_prediction.shape[1]
            y_true = self._to_one_hot(y_true, n_classes)

        y_prediction = np.clip(y_prediction, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_prediction), axis=1))
    
    def compute_gradient(self, X, y_true, y_prediction):
        if y_true.ndim == 1:
            n_classes = y_prediction.shape[1]
            y_true = self._to_one_hot(y_true, n_classes)

        m = len(y_true)
        grad_logits = (y_prediction - y_true) / m  # derivative of cross-entropy wrt logits
        dw = X.T @ grad_logits                     # (n_features, n_classes)
        db = np.sum(grad_logits, axis=0)           # (n_classes, )

        return np.concatenate([dw.flatten(), db.flatten()])
    
class HingeLoss(LossFunction):
    def __init__(self):
        self.C = 1.0
        self.weights = None

    def compute_loss(self, y_true, y_predictions):
        y_true = np.where(y_true >= 0, 1, -1)
        hinge_loss = np.mean(np.maximum(0, 1 - y_true * y_predictions))
        reg_loss = 0.5 * np.sum(self.weights ** 2) if self.weights is not None else 0.0
        
        return reg_loss + self.C * hinge_loss
    
    def compute_gradient(self, X, y_true, y_prediction):
        y_true = np.where(y_true >= 0, 1, -1)
        m = X.shape[0]

        margin = y_true * y_prediction
        mask = (margin < 1).astype(float)

        if self.weights is None:
            self.weights = np.zeros(X.shape[1])

        dw = self.weights - self.C * (X.T.dot(y_true * mask) / m)
        db = -self.C * np.sum(y_true * mask) / m

        return np.concatenate([dw.flatten(), [db]])
        