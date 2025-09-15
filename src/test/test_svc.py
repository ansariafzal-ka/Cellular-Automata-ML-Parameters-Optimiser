from src.models.svm_models import SupportVectorClassifier
from src.loss_functions.classification_losses import HingeLoss
from src.optimisers.gradient_based import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from src.utils import configurations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

model = SupportVectorClassifier(n_features=n_features)
hinge = HingeLoss()

batch_gradient_descent = BatchGradientDescent(configurations.ALPHA)
stochastic_gradient_descent = StochasticGradientDescent(configurations.ALPHA)
mini_batch_gradient_descent = MiniBatchGradientDescent(configurations.ALPHA)

## BATCH GRADIENT DESCENT (BGD)
print("--- Training with Batch Gradient Descent ---")
bgd_results = batch_gradient_descent.optimise(model, hinge, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(bgd_results["parameters"])

y_test_pred_bgd = model.predict(X_test)
report_bgd = classification_report(y_test, y_test_pred_bgd)
print("\nClassification Report for BGD:")
print(report_bgd)

## STOCHASTIC GRADIENT DESCENT (SGD)
print("--- Training with Stochastic Gradient Descent ---")
sgd_results = stochastic_gradient_descent.optimise(model, hinge, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(sgd_results["parameters"])

y_test_pred_sgd = model.predict(X_test)
report_sgd = classification_report(y_test, y_test_pred_sgd)
print("\nClassification Report for SGD:")
print(report_sgd)

## MINI-BATCH GRADIENT DESCENT (MBGD)
print("--- Training with Mini-Batch Gradient Descent ---")
mini_bgd_results = mini_batch_gradient_descent.optimise(model, hinge, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(mini_bgd_results["parameters"])

y_test_pred_mbgd = model.predict(X_test)
report_mbgd = classification_report(y_test, y_test_pred_mbgd)
print("\nClassification Report for MBGD:")
print(report_mbgd)