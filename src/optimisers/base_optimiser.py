from abc import ABC, abstractmethod

class Optimiser(ABC):

    @abstractmethod
    def optimise(self, model, loss_function, X, y, max_iters):
        pass