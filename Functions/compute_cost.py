import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    J = (1 / (2 * m)) * np.sum(sqrErrors)
    return J
