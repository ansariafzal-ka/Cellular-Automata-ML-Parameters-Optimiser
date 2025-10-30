from src.models.svm_models import SupportVectorRegression
from src.loss_functions.regression_losses import EpsilonIntensitiveLoss, MeanSquaredError, MeanAbsoluteError
from src.optimisers.gradient_based import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from src.optimisers.cellular_automata import CellularAutomataOptimiser
from src.utils import configurations

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


train_mse_results = []
test_mse_results = []
train_mae_results = []
test_mae_results = []
train_r2_results = []
test_r2_results = []

df = pd.read_csv('src/datasets/diabetes.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configurations.TEST_SIZE, random_state=configurations.SEED)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

n_features = X_train.shape[1]

model = SupportVectorRegression(n_features=n_features)
epsilon_intensive_loss = EpsilonIntensitiveLoss(model.C, model.weights, epsilon=0.1)
mse = MeanSquaredError()
mae = MeanAbsoluteError()

batch_gradient_descent = BatchGradientDescent(configurations.ALPHA)
stochastic_gradient_descent = StochasticGradientDescent(configurations.ALPHA)
mini_batch_gradient_descent = MiniBatchGradientDescent(configurations.ALPHA, batch_size=16)
ca_optimiser = CellularAutomataOptimiser(L=5, mu=0.01, omega=0.8)

## BATCH GRADIENT DESCENT (BGD)
bgd_results = batch_gradient_descent.optimise(model, epsilon_intensive_loss, X_train, y_train, max_iters=configurations.MAX_ITERS)
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
sgd_results = stochastic_gradient_descent.optimise(model, epsilon_intensive_loss, X_train, y_train, max_iters=configurations.MAX_ITERS)
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
mini_bgd_results = mini_batch_gradient_descent.optimise(model, epsilon_intensive_loss, X_train, y_train, max_iters=configurations.MAX_ITERS)
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

## CELLULAR AUTOMATA
model.set_param_bounds([(-1000.0, 1000.0)]) # for diabetes
# model.set_param_bounds([(-10.0, 10.0)]) # for california housing
ca_max_iters = 1000
ca_results = ca_optimiser.optimise(model, epsilon_intensive_loss, X_train, y_train, max_iters=ca_max_iters)
model.set_params(ca_results['parameters'])

y_train_pred_ca = model.predict(X_train)
y_test_pred_ca = model.predict(X_test)

train_mse_ca = mse.compute_loss(y_train, y_train_pred_ca)
test_mse_ca = mse.compute_loss(y_test, y_test_pred_ca)
train_mae_ca = mae.compute_loss(y_train, y_train_pred_ca)
test_mae_ca = mae.compute_loss(y_test, y_test_pred_ca)
train_r2_ca = r2_score(y_train, y_train_pred_ca)
test_r2_ca = r2_score(y_test, y_test_pred_ca)

train_mse_results.append(train_mse_ca)
test_mse_results.append(test_mse_ca)
train_mae_results.append(train_mae_ca)
test_mae_results.append(test_mae_ca)
train_r2_results.append(train_r2_ca)
test_r2_results.append(test_r2_ca)

## SKLEARN SVR
sk_model = SVR()
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
print(f"Configuration: MAX_ITERS={configurations.MAX_ITERS}, TEST_SIZE={configurations.TEST_SIZE}, ALPHA={configurations.ALPHA}, SEED={configurations.SEED}")
print(f"Cellular Automata Optimiser Configurations: L={ca_optimiser.L}, μ={ca_optimiser.mu}, ω={ca_optimiser.omega}, max_iters={ca_max_iters}")
print("="*50)

optimizers = ["Batch Gradient Descent", "Stochastic Gradient Descent", "Mini-Batch Gradient Descent", "Cellular Automata", "Sklearn SVR"]

for i, optimizer in enumerate(optimizers):
    print(f"--- {optimizer} ---")
    print(f"  Training MSE: {train_mse_results[i]:.4f}")
    print(f"  Testing MSE:  {test_mse_results[i]:.4f}")
    print(f"  Training MAE: {train_mae_results[i]:.4f}")
    print(f"  Testing MAE:  {test_mae_results[i]:.4f}")
    print(f"  Training r2 score: {train_r2_results[i]:.4f}")
    print(f"  Testing r2 score: {test_r2_results[i]:.4f}")
    print()