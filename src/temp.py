# test_ca_optimiser.py
import numpy as np
from optimisers.cellular_automata import CellularAutomataOptimiser
from models.linear_models import LinearRegression  # Replace with your actual model class
from loss_functions.regression_losses import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd

# Create sample data

# Sk learn params: [array([44.24418216]), np.float64(0.09922221422587896)]

print('Running...')

df = pd.read_csv('datasets/simple_linear_regression_data.csv')
X = df.drop("target", axis=1)
y = df["target"]

# Initialize optimiser and model
optimiser = CellularAutomataOptimiser(L=3)
model = LinearRegression(n_features=1)
mse = MeanSquaredError()


# Test one evaluation
best_params, best_loss = optimiser.optimise(model, mse, X, y, max_iters=5)

model.set_params(best_params)
y_pred_ca = model.predict(X)
manual_mse_ca = np.mean((y - y_pred_ca) ** 2)

# print(f"CA reported loss: {best_loss}")
# print(f"Manual MSE for CA params: {manual_mse_ca}")
# print(f"Best params: {best_params}")

# plt.scatter(X, y)
# x_line = np.linspace(X.min(), X.max(), 100)
# y_line = best_params[0] * x_line + best_params[1]

# plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Predicted: y={best_params[0]:.2f}x + {best_params[1]:.2f}')
# sk_slope = 44.24418216
# sk_intercept = 0.09922221422587896
# y_line_sk = sk_slope * x_line + sk_intercept
# plt.plot(x_line, y_line_sk, 'g--', linewidth=2, label=f'Sklearn: y={sk_slope:.2f}x + {sk_intercept:.2f}')
# plt.legend()
# plt.show()