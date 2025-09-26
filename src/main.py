# For iris dataset (3 classes)
from src.models.linear_models import SoftmaxRegression
from src.loss_functions.classification_losses import CategoricalCrossEntropy
from src.optimisers.gradient_based import BatchGradientDescent
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# Load iris data
df = pd.read_csv("src/datasets/custom.csv")
X = df.drop("target", axis=1).to_numpy()
y = df["target"].to_numpy()  # Should be [0, 1, 2, 0, 1, ...]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = SoftmaxRegression(n_features=5, n_classes=5)


# Use categorical cross-entropy
loss_fn = CategoricalCrossEntropy()

# Train
optimizer = BatchGradientDescent(learning_rate=0.01)
result = optimizer.optimise(model, loss_fn, X_train, y_train, max_iters=5000)

# Evaluate
model.set_params(result['parameters'])
# 1. Get probability predictions (shape (30, 3))
probability_predictions = model.predict(X_test)
# 2. Convert probabilities to hard class indices (shape (30,))
# argmax(axis=1) finds the column index (class) with the max probability for each row (sample).
class_predictions = probability_predictions.argmax(axis=1) 
# 3. Compare the 1D arrays (class_predictions (30,) == y_test (30,))
accuracy = np.mean(class_predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")

print("\n--- First 5 Labels ---")
print(f"True Labels (y_test[:5]): {y_test[:10]}")
print(f"Predicted Labels (class_predictions[:5]): {class_predictions[:10]}")

print(classification_report(y_test, class_predictions))