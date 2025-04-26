import numpy as np

def compute_cost(X, y, theta):
    """Computa a função de custo J para a regressão linear."""
    m = len(y)  # número de exemplos de treinamento
    predictions = X.dot(theta)
    errors = predictions - y
    J = (1 / (2 * m)) * np.dot(errors.T, errors)
    return J
