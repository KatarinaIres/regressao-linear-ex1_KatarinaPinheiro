import matplotlib.pyplot as plt

def plot_data(X, y):
    """Plota os dados X e y em um gráfico de dispersão."""
    plt.figure()
    plt.plot(X, y, 'rx', markersize=5)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter plot of training data')
    plt.grid(True)
    plt.show() 