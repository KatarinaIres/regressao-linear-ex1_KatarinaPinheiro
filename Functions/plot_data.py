import matplotlib.pyplot as plt

def plot_data(X, y, color='red', marker='x', label=None):
    """
    Plota os dados X e y em um gráfico de dispersão.
    
    Parâmetros:
    - X: array de características (feature)
    - y: array de saídas (target)
    - color: cor dos pontos (default 'red')
    - marker: formato do marcador (default 'x')
    - label: legenda opcional
    """
    plt.scatter(X, y, c=color, marker=marker, s=40, label=label)
    plt.grid(True)
