import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def animate_gradient_descent(X, y, theta_history):
    fig, ax = plt.subplots()
    
    ax.scatter(X[:, 1], y, color='blue', label='Dados de treinamento')
    line, = ax.plot([], [], 'r-', linewidth=2, label='Regressão Linear')
    
    ax.set_xlabel('População da cidade')
    ax.set_ylabel('Lucro')
    ax.set_title('Descida do Gradiente')
    ax.legend()
    ax.grid(True)

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        current_theta = theta_history[i]
        y_pred = np.dot(X, current_theta)
        line.set_data(X[:, 1], y_pred)
        return (line,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(theta_history),
        init_func=init, blit=True, interval=50
    )

    # Cria pasta de Figures se não existir
    os.makedirs('Figures', exist_ok=True)

    # Salva o gif diretamente
    anim.save('Figures/animacao_descida_gradiente.gif', writer='pillow')

    plt.close(fig)
