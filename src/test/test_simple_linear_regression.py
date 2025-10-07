from src.models.linear_models import LinearRegression
from src.loss_functions.regression_losses import MeanSquaredError, MeanAbsoluteError
from src.optimisers.gradient_based import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from src.utils import configurations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SK_LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

n_features = 1
train_mse_results = []
test_mse_results = []
train_mae_results = []
test_mae_results = []
train_r2_results = []
test_r2_results = []

df = pd.read_csv("src/datasets/simple_linear_regression_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configurations.TEST_SIZE, random_state=configurations.SEED)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = LinearRegression(n_features=n_features)
mse = MeanSquaredError()
mae = MeanAbsoluteError()

batch_gradient_descent = BatchGradientDescent(configurations.ALPHA)
stochastic_gradient_descent = StochasticGradientDescent(configurations.ALPHA)
mini_batch_gradient_descent = MiniBatchGradientDescent(configurations.ALPHA)

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

## SKLEARN SIMPLE LINEAR REGRESSION
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

# Add to your results lists
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


# Plot convergence for all methods
# plt.figure(figsize=(12, 8))

# plt.plot(bgd_results['loss_history'], label='Batch GD', alpha=0.7)
# plt.plot(sgd_results['loss_history'], label='Stochastic GD', alpha=0.7)
# plt.plot(mini_bgd_results['loss_history'], label='Mini-Batch GD', alpha=0.7)

# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Convergence Comparison of Gradient Descent Variants')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(X_train, y_train, label='Actual Data')
ax[0].plot(X_train, y_train_pred_bgd, color='red', label='Regression Line')
ax[0].set_title("mbgd - Training Data")
ax[0].set_xlabel("X_train")
ax[0].set_ylabel("y_train")
ax[0].legend()

ax[1].scatter(X_test, y_test, label='Actual Data')
ax[1].plot(X_test, y_test_pred_mbgd, color='red', label='Regression Line')
ax[1].set_title("mbgd - Testing Data")
ax[1].set_xlabel("X_test")
ax[1].set_ylabel("y_test")
ax[1].legend()

plt.tight_layout()
plt.show()