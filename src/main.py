import numpy as np
import matplotlib.pyplot as plt
from models.svm_models import SupportVectorClassifier

# This script assumes the SupportVectorClassifier class is defined in a separate file or module.
# The class must have the following methods:
# - __init__(self, n_features)
# - predict(self, X)
# - set_params(self, params)
# - get_param_count(self)

if __name__ == "__main__":
    # 1. Generate dummy, linearly separable data
    np.random.seed(42)
    n_samples = 100
    n_features = 2

    # Define two clusters that are easily separable by a line
    X_class1 = np.random.randn(n_samples // 2, n_features) - np.array([2, 2])
    y_class1 = -np.ones(n_samples // 2)

    X_class2 = np.random.randn(n_samples // 2, n_features) + np.array([2, 2])
    y_class2 = np.ones(n_samples // 2)

    X = np.vstack([X_class1, X_class2])
    y = np.hstack([y_class1, y_class2])
    
    # 2. Define the three new test points.
    test_points = np.array([
        [4.0, 5.0],    # Point 1: Clearly on the positive side (red)
        [-3.0, -4.0],  # Point 2: Clearly on the negative side (blue)
        [0.1, -0.1]    # Point 3: Very close to the boundary, on the negative side
    ])

    # 3. Initialize the model (assuming the class is imported)
    try:
        from models.svm_models import SupportVectorClassifier
        svc = SupportVectorClassifier(n_features=n_features)
    except (NameError, ImportError):
        print("Error: SupportVectorClassifier class not found.")
        print("A simplified version will be used for demonstration.")
        
        # A simplified version of the SupportVectorClassifier class
        class SupportVectorClassifier:
            def __init__(self, n_features):
                self.n_features = n_features
                self.weights = np.zeros(n_features)
                self.bias = 0.0

            def set_params(self, params):
                self.weights = params[:-1]
                self.bias = params[-1]

            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                prediction_values = np.dot(X, self.weights) + self.bias
                return np.sign(prediction_values)

        svc = SupportVectorClassifier(n_features=n_features)

    # 4. Manually set known optimal parameters
    optimal_weights = np.array([0.9, 0.9])
    optimal_bias = 0.0
    
    optimal_params = np.concatenate([optimal_weights, [optimal_bias]])
    svc.set_params(optimal_params)

    # 5. Get predictions for the new test points
    test_predictions = svc.predict(test_points)

    # 6. Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    # Plot the original data points, colored by class
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k', label='Training Data')
    
    # Plot the decision boundary and the margins
    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 10)
    yy = -(optimal_weights[0] / optimal_weights[1]) * xx - optimal_bias / optimal_weights[1]
    
    yy_plus = -(optimal_weights[0] / optimal_weights[1]) * xx - (optimal_bias - 1) / optimal_weights[1]
    yy_minus = -(optimal_weights[0] / optimal_weights[1]) * xx - (optimal_bias + 1) / optimal_weights[1]
    
    plt.plot(xx, yy, 'k-', label='Decision Boundary')
    plt.plot(xx, yy_plus, 'k--', label='Margin')
    plt.plot(xx, yy_minus, 'k--')

    # Plot the new test points with a distinct marker and color
    # The colors are based on the predictions
    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_predictions, cmap='coolwarm',
                marker='*', s=300, edgecolors='black', linewidths=2, label='New Test Points')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Support Vector Classifier: Decision Boundary and New Point Predictions')
    
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the predictions to the console
    print("--- Predictions for new data points ---")
    for i, point in enumerate(test_points):
        predicted_class = "Positive (+1)" if test_predictions[i] > 0 else "Negative (-1)"
        print(f"Point: {point} -> Predicted Class: {predicted_class}")