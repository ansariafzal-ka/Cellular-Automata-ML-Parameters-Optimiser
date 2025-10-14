from src.models.linear_models import LogisticRegression
from src.loss_functions.classification_losses import BinaryCrossEntropy
from src.optimisers.gradient_based import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from src.optimisers.cellular_automata import CellularAutomataOptimiser
from src.utils import configurations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(configurations.SEED)

train_results = []
test_results = []

df = pd.read_csv("src/datasets/breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configurations.TEST_SIZE, random_state=configurations.SEED)

n_features = X_train.shape[1]

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = LogisticRegression(n_features=n_features)
bce = BinaryCrossEntropy()

batch_gradient_descent = BatchGradientDescent(configurations.ALPHA)
stochastic_gradient_descent = StochasticGradientDescent(configurations.ALPHA)
mini_batch_gradient_descent = MiniBatchGradientDescent(configurations.ALPHA)

print("="*50)
print("             OPTIMIZATION ALGORITHMS RESULTS")
print("="*50)
print(f"Configuration: MAX_ITERS={configurations.MAX_ITERS}, TEST_SIZE={configurations.TEST_SIZE}, ALPHA={configurations.ALPHA}, SEED={configurations.SEED}")
print("="*50)

## BATCH GRADIENT DESCENT (BGD)
print("--- Training with Batch Gradient Descent ---")
bgd_results = batch_gradient_descent.optimise(model, bce, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(bgd_results["parameters"])

y_test_pred_bgd = model.predict_labels(X_test)
report_bgd = classification_report(y_test, y_test_pred_bgd)
print("\nClassification Report for BGD:")
print(report_bgd)

## STOCHASTIC GRADIENT DESCENT (SGD)
print("--- Training with Stochastic Gradient Descent ---")
sgd_results = stochastic_gradient_descent.optimise(model, bce, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(sgd_results["parameters"])

y_test_pred_sgd = model.predict_labels(X_test)
report_sgd = classification_report(y_test, y_test_pred_sgd)
print("\nClassification Report for SGD:")
print(report_sgd)

## MINI-BATCH GRADIENT DESCENT (MBGD)
print("--- Training with Mini-Batch Gradient Descent ---")
mini_bgd_results = mini_batch_gradient_descent.optimise(model, bce, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(mini_bgd_results["parameters"])

y_test_pred_mbgd = model.predict_labels(X_test)
report_mbgd = classification_report(y_test, y_test_pred_mbgd)
print("\nClassification Report for MBGD:")
print(report_mbgd)

## CELLULAR AUTOMATA
model.set_param_bounds([(-1000.0, 1000.0)]) # gives good result although it gives overflow warning
cellular_automata = CellularAutomataOptimiser(L=5, mu=0.01, omega=0.8)
cellular_automata_results = cellular_automata.optimise(model, bce, X_train, y_train, max_iters=10)
model.set_params(cellular_automata_results["parameters"])

y_test_pred_cellular_automata = model.predict_labels(X_test)
report_cellular_automata = classification_report(y_test, y_test_pred_cellular_automata)
print("\nClassification Report for Cellular Automata:")
print(report_cellular_automata)

## SkLEARN LOGISTIC REGRESSION
sk_model = SKLogisticRegression()
sk_model.fit(X_train, y_train)

# sk_params = [sk_model.coef_, sk_model.intercept_]
# print(f"sk params : {sk_params}")

y_test_pred_sk = sk_model.predict(X_test)
print("\nClassification Report for Sklearn:")
print(classification_report(y_test, y_test_pred_sk))