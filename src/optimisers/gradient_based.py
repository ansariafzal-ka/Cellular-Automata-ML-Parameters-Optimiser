from .base_optimiser import Optimiser
import numpy as np

class BatchGradientDescent(Optimiser):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def optimise(self, model, loss_function, X, y, max_iters=1000):
        param_bounds = model.get_param_bounds()
        params = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in param_bounds])
        model.set_params(params)

        loss_history = []

        for _ in range(max_iters):
            predictions = model.predict(X)
            loss = loss_function.compute_loss(y, predictions)
            loss_history.append(loss)

            gradients = loss_function.compute_gradient(X, y, predictions)
            params = params - self.learning_rate * gradients
            model.set_params(params)

        return {"algorithm": "BatchGradientDescent","parameters": params, "best_loss": loss_history[-1], "loss_history": loss_history}
    
class StochasticGradientDescent(Optimiser):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimise(self, model, loss_function, X, y, max_iters):
        param_bounds = model.get_param_bounds()
        params = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in param_bounds])
        model.set_params(params)

        loss_history = []
        m = len(X)

        for _ in range(max_iters):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(m):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]

                predictions = model.predict(xi)
                loss = loss_function.compute_loss(yi, predictions)
                epoch_loss += loss

                gradients = loss_function.compute_gradient(xi, yi, predictions)
                params = params - self.learning_rate * gradients
                model.set_params(params)
            
            avg_loss = epoch_loss / m
            loss_history.append(avg_loss)
        
        return {"algorithm": "StochasticGradientDescent","parameters": params, "best_loss": loss_history[-1], "loss_history": loss_history}
    

class MiniBatchGradientDescent(Optimiser):
    def __init__(self, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def optimise(self, model, loss_function, X, y, max_iters):
        param_bounds = model.get_param_bounds()
        params = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in param_bounds])
        model.set_params(params)

        loss_history = []
        m = len(X)

        for _ in range(max_iters):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, m, self.batch_size):
                end_index = min(i+self.batch_size, m)
                X_batch = X_shuffled[i:end_index]
                y_batch = y_shuffled[i:end_index]

                predictions = model.predict(X_batch)
                loss = loss_function.compute_loss(y_batch, predictions)
                epoch_loss += loss
                n_batches += 1

                gradients = loss_function.compute_gradient(X_batch, y_batch, predictions)
                params = params - self.learning_rate * gradients
                model.set_params(params)

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

        return {"algorithm": "MiniBatchGradientDescent","parameters": params, "best_loss": loss_history[-1], "loss_history": loss_history}

