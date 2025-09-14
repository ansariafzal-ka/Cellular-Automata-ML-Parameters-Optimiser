from models.linear_models import LinearRegression, LogisticRegression, PolynomialRegression
from loss_functions.regression_losses import MeanSquaredError, MeanAbsoluteError
from sklearn.linear_model import LinearRegression as sk_LinearRegression
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# datasets
df_reg = pd.read_csv("./datasets/california_housing.csv")
df_poly = pd.read_csv("./datasets/polynomial_data.csv")
df_classification = pd.read_csv("./datasets/breast_cancer.csv")

X = df_reg.drop("target", axis=1)
y = df_reg["target"]

sk_lr = sk_LinearRegression()

# training
sk_lr.fit(X, y)
sk_lr_pred = sk_lr.predict(X)

# get params from sklean model
sk_lr_weights = sk_lr.coef_
sk_lr_bias = sk_lr.intercept_
sk_lr_params = np.concatenate([sk_lr_weights, [sk_lr_bias]])

lr = LinearRegression(n_features=8)
lr.set_params(sk_lr_params)
lr_pred = lr.predict(X)

# compute loss
mse = MeanSquaredError()
mae = MeanAbsoluteError()

print("\n Linear Regression")
print(X.shape)
print(f"Scikit-learn Mean Squared Error: {mse.compute_loss(y, sk_lr_pred)}")
print(f"Custom Model Mean Squared Error: {mse.compute_loss(y, lr_pred)}")
print(f"Scikit-learn Mean Absolute Error: {mae.compute_loss(y, sk_lr_pred)}")
print(f"Custom Model Mean Absolute Error: {mae.compute_loss(y, lr_pred)}")
print("-" * 40)

X_poly = df_poly.drop("target", axis=1)
y_poly = df_poly["target"]
poly_degree = 3

sk_poly_lr = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), sk_LinearRegression())
sk_poly_lr.fit(X_poly, y_poly)
sk_poly_pred = sk_poly_lr.predict(X_poly)

sk_poly_weights = sk_poly_lr.steps[1][1].coef_
sk_poly_bias = sk_poly_lr.steps[1][1].intercept_
sk_poly_params = np.concatenate([sk_poly_weights, [sk_poly_bias]])

poly_reg = PolynomialRegression(n_features=X_poly.shape[1], degree=poly_degree)
poly_reg.set_params(sk_poly_params)
poly_reg_pred = poly_reg.predict(X_poly)

print("\n Polynomial Regression")
print(X_poly.shape)
print(f"Scikit-learn Mean Squared Error: {mse.compute_loss(y_poly, sk_poly_pred)}")
print(f"Custom Model Mean Squared Error: {mse.compute_loss(y_poly, poly_reg_pred)}")
print(f"Scikit-learn Mean Absolute Error: {mae.compute_loss(y_poly, sk_poly_pred)}")
print(f"Custom Model Mean Absolute Error: {mae.compute_loss(y_poly, poly_reg_pred)}")
print("-" * 40)

X_cls = df_classification.drop("target", axis=1)
y_cls = df_classification["target"]

# sklearn logistic regression (binary for now, so filter classes)
# Example: keep only two classes (say 0 and 1)
binary_mask = y_cls.isin([0, 1])
sk_log_reg = sk_LogisticRegression()
sk_log_reg.fit(X_cls, y_cls)
sk_log_pred = sk_log_reg.predict(X_cls)

# get params from sklearn model
sk_log_weights = sk_log_reg.coef_[0]   # only one row for binary
sk_log_bias = sk_log_reg.intercept_[0]
sk_log_params = np.concatenate([sk_log_weights, [sk_log_bias]])

# custom logistic regression
log_reg = LogisticRegression(n_features=X_cls.shape[1])
log_reg.set_params(sk_log_params)
log_reg_pred = log_reg.predict(X_cls)

# compare accuracies
print("\n Logistic Regression (Binary)")
print(X_cls.shape)
print(f"Scikit-learn Accuracy: {accuracy_score(y_cls, sk_log_pred)}")
print(f"Custom Model Accuracy: {accuracy_score(y_cls, log_reg_pred)}")
print("-" * 40)