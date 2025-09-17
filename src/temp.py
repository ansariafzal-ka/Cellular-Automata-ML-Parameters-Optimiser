from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from loss_functions.regression_losses import MeanSquaredError, MeanAbsoluteError
import pandas as pd

df = pd.read_csv("./datasets/california_housing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = MeanSquaredError()
mae = MeanAbsoluteError()

print(f"MSE: {mse.compute_loss(y_test, y_pred)}")
print(f"MAE: {mae.compute_loss(y_test, y_pred)}")
print(f"r2 score: {r2_score(y_test, y_pred)}")