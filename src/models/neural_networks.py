from .base_model import Model
import numpy as np

class MultiLayerPerceptron(Model):

    def __init__(self, n_features, n_hidder_layers, activation="relu"):
        self.n_features = n_features
        self.n_hidden_layers = n_hidder_layers
        self.activation = self._get_activation_function(activation)
        self.layers = []
        self._initialise_layers()

    def _get_activation_function(self, activation):

        if activation == "relu":
            return lambda z: np.maximum(0, z)
        elif activation == "sigmoid":
            return lambda z: 1 / (1 + np.exp(-z))
        else:
            raise ValueError(f"Activation function {activation} not supported.")
        
    def _initialise_layers(self):
        layer_sizes = [self.n_features] + self.n_hidden_layers + [1]
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            weights = np.zeros((n_in, n_out))
            bias = np.zeros(n_out)
            self.layers.append({"weights": weights, "bias": bias})

    def predict(self, X):
        z = X.T
        for i, layer in enumerate(self.layers):
            weights = layer["weights"]
            bias = layer["bias"].reshape(-1, 1)
            z = weights @ z + bias
            if i < len(self.layers) - 1:
                z = self.activation(z)

        return z.T.flatten()
    
    def set_params(self, params):
        start = 0
        for layer in self.layers:
            weights_size = layer["weights"].size
            weights_shape = layer["weights"].shape
            layer["weights"] = params[start:weights_size+start].reshape(weights_shape)
            start += weights_size

            bias_size = layer["bias"].size

            layer["bias"] = params[start:start + bias_size]
            start += bias_size

    def get_param_count(self):
        count = 0
        for layer in self.layers:
            count += layer["weights"].size
            count += layer["bias"].size
        return count

    def get_params(self):
        params = []
        for layer in self.layers:
            params.append(layer["weights"].flatten())
            params.append(layer["bias"])
        return np.concatenate(params)
    
    def get_param_bounds(self):
        num_params = self.get_param_count()
        return [(-5, 5)] * num_params
