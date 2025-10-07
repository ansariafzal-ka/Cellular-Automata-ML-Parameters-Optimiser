# test_ca_optimiser.py
import numpy as np
from optimisers.cellular_automata import CellularAutomataOptimiser
from models.linear_models import LinearRegression  # Replace with your actual model class
from loss_functions.regression_losses import MeanSquaredError
import matplotlib.pyplot as plt

# Create sample data

X = np.random.rand(100, 1)
slope = 2.5
intercept = 1.0
y = slope * X + intercept + 0.1 * np.random.randn(100, 1)

# Initialize optimiser and model
optimiser = CellularAutomataOptimiser(L=3)
model = LinearRegression(n_features=1)
mse = MeanSquaredError()

# Test one evaluation
best_params, best_loss = optimiser.optimise(model, mse, X, y, max_iters=1)
print(f"Best loss: {best_loss}")
print(f"Best params: {best_params}")

# plt.scatter(X, y)
# plt.show()