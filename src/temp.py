# test_ca_optimiser.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.metrics import classification_report

from optimisers.cellular_automata import CellularAutomataOptimiser
from models.linear_models import LogisticRegression
from loss_functions.classification_losses import BinaryCrossEntropy

# Load data
df = pd.read_csv('datasets/breast_cancer.csv')
X = df.drop("target", axis=1)
y = df["target"]

# Initialize optimiser and model
optimiser = CellularAutomataOptimiser(L=5, mu=0.01, omega=0.8)
logistic_model = LogisticRegression(n_features=X.shape[1])
bce = BinaryCrossEntropy()

# Sklearn logistic regression for comparison
sk_model = SKLogistic()
sk_model.fit(X, y)
y_pred_sk = sk_model.predict(X)

print("=" * 50)
print("SKLEARN LOGISTIC REGRESSION")
print("=" * 50)
print(classification_report(y, y_pred_sk))

# Print sklearn parameters
print("\nSklearn Parameters:")
print(f"Coefficients: {sk_model.coef_[0]}")
print(f"Intercept: {sk_model.intercept_[0]}")

# Cellular Automata optimization
best_params, best_loss = optimiser.optimise(logistic_model, bce, X, y, max_iters=10)
logistic_model.set_params(best_params)

# Get probabilities and convert to predictions
y_probs_ca = logistic_model.predict(X)
y_pred_ca = (y_probs_ca > 0.5).astype(int)

print("\n" + "=" * 50)
print("CELLULAR AUTOMATA LOGISTIC REGRESSION")
print("=" * 50)
print(classification_report(y, y_pred_ca))

# Print CA parameters
print(f"\nCellular Automata Parameters:")
print(f"Best Loss: {best_loss:.6f}")
print(f"Parameters: {best_params}")