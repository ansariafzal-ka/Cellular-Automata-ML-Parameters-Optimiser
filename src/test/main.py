import pandas as pd
import numpy as np

from src.models.svm_models import SupportVectorClassifier
from src.loss_functions.classification_losses import HingeLoss
from src.optimisers.gradient_based import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from src.optimisers.cellular_automata import CellularAutomataOptimiser
from src.utils import configurations

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# np.random.seed(configurations.SEED)

df = pd.read_csv('src/datasets/breast_cancer.csv')

X = df.drop('target', axis=1)
y = df['target']

y = np.where(y > 0, 1, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configurations.TEST_SIZE, random_state=configurations.SEED)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

n_features = X_train.shape[1]
model = SupportVectorClassifier(n_features=n_features)

hinge_loss = HingeLoss()

batch_gradient_descent = BatchGradientDescent(configurations.ALPHA)
stochastic_gradient_descent = StochasticGradientDescent(configurations.ALPHA)
mini_batch_gradient_descent = MiniBatchGradientDescent(configurations.ALPHA)

ca_optimiser = CellularAutomataOptimiser(L=5, mu=0.01, omega=0.8)
ca_max_iters = 1000

## BATCH GRADIENT DESCENT ##
bgd_results = batch_gradient_descent.optimise(model, model.loss_fn, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(bgd_results["parameters"])

y_test_pred_bgd = model.predict(X_test)

print('='*50)
print('BGD Classification Report:')
print(classification_report(y_test, y_test_pred_bgd))
print('='*50)


## STOCHASTIC GRADIENT DESCENT ##
sgd_results = stochastic_gradient_descent.optimise(model, model.loss_fn, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(sgd_results["parameters"])

y_test_pred_sgd = model.predict(X_test)

print('='*50)
print('SGD Classification Report:')
print(classification_report(y_test, y_test_pred_sgd))
print('='*50)

## MINI BATCH GRADIENT DESCENT ##
mini_bgd_results = mini_batch_gradient_descent.optimise(model, model.loss_fn, X_train, y_train, max_iters=configurations.MAX_ITERS)
model.set_params(mini_bgd_results["parameters"])

y_test_pred_mbgd = model.predict(X_test)

print('='*50)
print('Mini BGD Classification Report:')
print(classification_report(y_test, y_test_pred_mbgd))
print('='*50)

## CELLULAR AUTOMATA ##
model.set_param_bounds([(-3.0, 3.0)])
cellular_automata_results = ca_optimiser.optimise(model, model.loss_fn, X_train, y_train, max_iters=1000)
# print(f"cellular automata params: {cellular_automata_results['parameters']}")
model.set_params(cellular_automata_results["parameters"])

y_test_pred_cellular_automata = model.predict(X_test)

print('='*50)
print("CA Classification Report:")
print(classification_report(y_test, y_test_pred_bgd))
print('='*50)

## SKLEARN MODEL ##
sk_model = SVC(kernel='linear')
sk_model.fit(X_train, y_train)
sk_predictions = sk_model.predict(X_test)

print('='*50)
print('SK Model Classification Report:')
print(classification_report(y_test, sk_predictions))
print('='*50)

# print(f'Sk learn parameters: {sk_model.coef_, sk_model.intercept_}')