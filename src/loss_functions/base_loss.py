from abc import ABC, abstractmethod

class LossFunction(ABC):

    @abstractmethod
    def compute_loss(self, y_true, y_predictions):
        pass

    @abstractmethod
    def compute_gradient(self, X, y_true, y_predictions):
        pass