import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from optimisers.gradient_based import StochasticGradientDescent
from models.svm_models import SupportVectorClassifier
from models.linear_models import LogisticRegression
from loss_functions.classification_losses import HingeLoss, BinaryCrossEntropy

# Results storage
results = []

# --- Data Preparation ---
df = pd.read_csv("./datasets/breast_cancer.csv")
X = df.drop("target", axis=1).to_numpy()
y = df["target"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Support Vector Classifier Test ---
print("=== Support Vector Classifier Test ===")
model_svm = SupportVectorClassifier(n_features=X_train.shape[1])
loss_function_svm = HingeLoss()
optimizer_svm = StochasticGradientDescent(learning_rate=0.01)

training_results_svm = optimizer_svm.optimise(
    model_svm, loss_function_svm, X_train, y_train, max_iters=1000
)

model_svm.set_params(training_results_svm['parameters'])
test_predictions_svm = model_svm.predict(X_test)
test_loss_svm = loss_function_svm.compute_loss(y_test, test_predictions_svm)

binary_predictions_svm = (test_predictions_svm >= 0).astype(int)

print(f"Training Loss: {training_results_svm['best_loss']:.4f}")
print(f"Test Loss: {test_loss_svm:.4f}")
print("Custom SVM weights:", model_svm.get_params()[:-1])
print("Custom SVM bias:", model_svm.get_params()[-1])
print("\n--- Classification Report (SVM) ---")
print(classification_report(y_test, binary_predictions_svm))

# --- Logistic Regression Test ---
print("=== Logistic Regression Test ===")
model_lr = LogisticRegression(n_features=X_train.shape[1])
loss_function_lr = BinaryCrossEntropy()
optimizer_lr = StochasticGradientDescent(learning_rate=0.01)

training_results_lr = optimizer_lr.optimise(
    model_lr, loss_function_lr, X_train, y_train, max_iters=1000
)

model_lr.set_params(training_results_lr['parameters'])
test_predictions_lr = model_lr.predict(X_test)
test_loss_lr = loss_function_lr.compute_loss(y_test, test_predictions_lr)

binary_predictions_lr = (test_predictions_lr >= 0.5).astype(int)

print(f"Training Loss: {training_results_lr['best_loss']:.4f}")
print(f"Test Loss: {test_loss_lr:.4f}")
print("Custom LR weights:", model_lr.get_params()[:-1])
print("Custom LR bias:", model_lr.get_params()[-1])
print("\n--- Classification Report (Logistic Regression) ---")
print(classification_report(y_test, binary_predictions_lr))