# test_ca_optimiser.py
import numpy as np
from optimisers.cellular_automata import CellularAutomataOptimiser
from models.linear_models import LinearRegression  # Replace with your actual model class
from sklearn.linear_model import LinearRegression as SKLinearRegression
from loss_functions.regression_losses import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv('datasets/simple_linear_regression_data.csv')
# X = df.drop("target", axis=1)
# y = df["target"]

X = np.random.rand(100, 1) * 10
slope = 2.5
bias = 1.0
y = slope * X + bias

# Initialize optimiser and model
optimiser = CellularAutomataOptimiser(L=5)
model = LinearRegression(n_features=1)
mse = MeanSquaredError()

sk_model = SKLinearRegression()
sk_model.fit(X, y)
sk_pred = sk_model.predict(X)


# Test one evaluation
best_params, best_loss = optimiser.optimise(model, mse, X, y, max_iters=10)

model.set_params(best_params)
y_pred_ca = model.predict(X)
manual_mse_ca = np.mean((y - y_pred_ca) ** 2)

# plot the line
# plt.scatter(X, y)
# plt.plot(X, y_pred_ca, color='red', label='ca line')
# plt.plot(X, sk_pred, color='green', label='sk line')
# plt.legend()
# plt.show()