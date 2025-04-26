import matplotlib.pyplot as plt

def plot_data(X, y):
    """Plota os dados X e y em um gráfico de dispersão."""
    plt.plot(X, y, 'rx', markersize=5)
    plt.xlabel('População da cidade')
    plt.ylabel('Lucro')
    plt.title('Dados de treinamento')
    plt.grid(True)
