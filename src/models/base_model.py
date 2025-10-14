from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_param_bounds(self, bounds):
        pass

    @abstractmethod
    def get_param_bounds(self):
        pass

    @abstractmethod
    def get_param_count(self):
        pass