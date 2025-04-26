import numpy as np

def gradient_descent(X, y, theta, alpha, num_iters):
    """Executa o gradiente descendente para aprender theta."""
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (X.T.dot(errors))
        cost = (1 / (2 * m)) * np.dot(errors.T, errors)
        J_history.append(cost)

    return theta, J_history
