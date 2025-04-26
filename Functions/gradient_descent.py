import numpy as np
from Functions.compute_cost import compute_cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    theta_history = [theta.copy()]

    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (X.T.dot(errors))
        J_history.append(compute_cost(X, y, theta))
        theta_history.append(theta.copy())  # Salva a trajetória

    return theta, J_history, theta_history
