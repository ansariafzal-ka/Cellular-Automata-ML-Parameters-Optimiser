from src.models.linear_models import LinearRegression
from src.loss_functions.regression_losses import MeanSquaredError, MeanAbsoluteError
from src.optimisers.gradient_based import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from src.utils import configurations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SK_LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

train_mse_results = []
test_mse_results = []
train_mae_results = []
test_mae_results = []
train_r2_results = []
test_r2_results = []

df = pd.read_csv("src/datasets/diabetes.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configurations.TEST_SIZE, random_state=configurations.SEED)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

n_features = X_train.shape[1]

model = LinearRegression(n_features=n_features)
mse = MeanSquaredError()
mae = MeanAbsoluteError()

batch_gradient_descent = BatchGradientDescent(configurations.ALPHA)
stochastic_gradient_descent = StochasticGradientDescent(configurations.ALPHA)
mini_batch_gradient_descent = MiniBatchGradientDescent(configurations.ALPHA, batch_size=16)

## BATCH GRADIENT DESCENT (BGD)
bgd_results = batch_gradient_descent.optimise(model, mse, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(bgd_results["parameters"])

y_train_pred_bgd = model.predict(X_train)
y_test_pred_bgd = model.predict(X_test)

train_mse_bgd = mse.compute_loss(y_train, y_train_pred_bgd)
test_mse_bgd = mse.compute_loss(y_test, y_test_pred_bgd)
train_mae_bgd = mae.compute_loss(y_train, y_train_pred_bgd)
test_mae_bgd = mae.compute_loss(y_test, y_test_pred_bgd)
train_r2_bgd = r2_score(y_train, y_train_pred_bgd)
test_r2_bgd = r2_score(y_test, y_test_pred_bgd)

train_mse_results.append(train_mse_bgd)
test_mse_results.append(test_mse_bgd)
train_mae_results.append(train_mae_bgd)
test_mae_results.append(test_mae_bgd)
train_r2_results.append(train_r2_bgd)
test_r2_results.append(test_r2_bgd)

## STOCHASTIC GRADIENT DESCENT (SGD)
sgd_results = stochastic_gradient_descent.optimise(model, mse, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(sgd_results["parameters"])

y_train_pred_sgd = model.predict(X_train)
y_test_pred_sgd = model.predict(X_test)

train_mse_sgd = mse.compute_loss(y_train, y_train_pred_sgd)
test_mse_sgd = mse.compute_loss(y_test, y_test_pred_sgd)
train_mae_sgd = mae.compute_loss(y_train, y_train_pred_sgd)
test_mae_sgd = mae.compute_loss(y_test, y_test_pred_sgd)
train_r2_sgd = r2_score(y_train, y_train_pred_sgd)
test_r2_sgd = r2_score(y_test, y_test_pred_sgd)

train_mse_results.append(train_mse_sgd)
test_mse_results.append(test_mse_sgd)
train_mae_results.append(train_mae_sgd)
test_mae_results.append(test_mae_sgd)
train_r2_results.append(train_r2_sgd)
test_r2_results.append(test_r2_sgd)

## MINI-BATCH GRADIENT DESCENT (MBGD)
mini_bgd_results = mini_batch_gradient_descent.optimise(model, mse, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(mini_bgd_results["parameters"])

y_train_pred_mbgd = model.predict(X_train)
y_test_pred_mbgd = model.predict(X_test)

train_mse_mbgd = mse.compute_loss(y_train, y_train_pred_mbgd)
test_mse_mbgd = mse.compute_loss(y_test, y_test_pred_mbgd)
train_mae_mbgd = mae.compute_loss(y_train, y_train_pred_mbgd)
test_mae_mbgd = mae.compute_loss(y_test, y_test_pred_mbgd)
train_r2_mbgd = r2_score(y_train, y_train_pred_mbgd)
test_r2_mbgd = r2_score(y_test, y_test_pred_mbgd)

train_mse_results.append(train_mse_mbgd)
test_mse_results.append(test_mse_mbgd)
train_mae_results.append(train_mae_mbgd)
test_mae_results.append(test_mae_mbgd)
train_r2_results.append(train_r2_mbgd)
test_r2_results.append(test_r2_mbgd)

## SKLEARN LINEAR REGRESSION
sk_model = SK_LinearRegression()
sk_model.fit(X_train, y_train)

y_train_pred_sk = sk_model.predict(X_train)
y_test_pred_sk = sk_model.predict(X_test)

train_mse_sk = mse.compute_loss(y_train, y_train_pred_sk)
test_mse_sk = mse.compute_loss(y_test, y_test_pred_sk)
train_mae_sk = mae.compute_loss(y_train, y_train_pred_sk)
test_mae_sk = mae.compute_loss(y_test, y_test_pred_sk)
train_r2_sk = r2_score(y_train, y_train_pred_sk)
test_r2_sk = r2_score(y_test, y_test_pred_sk)

train_mse_results.append(train_mse_sk)
test_mse_results.append(test_mse_sk)
train_mae_results.append(train_mae_sk)
test_mae_results.append(test_mae_sk)
train_r2_results.append(train_r2_sk)
test_r2_results.append(test_r2_sk)


print("="*50)
print("             OPTIMIZATION ALGORITHMS RESULTS")
print("="*50)

optimizers = ["Batch Gradient Descent", "Stochastic Gradient Descent", "Mini-Batch Gradient Descent", "Sklearn LinearRegression"]

for i, optimizer in enumerate(optimizers):
    print(f"--- {optimizer} ---")
    print(f"  Training MSE: {train_mse_results[i]:.4f}")
    print(f"  Testing MSE:  {test_mse_results[i]:.4f}")
    print(f"  Training MAE: {train_mae_results[i]:.4f}")
    print(f"  Testing MAE:  {test_mae_results[i]:.4f}")
    print(f"  Training r2 score: {train_r2_results[i]:.4f}")
    print(f"  Testing r2 score: {test_r2_results[i]:.4f}")
    print()