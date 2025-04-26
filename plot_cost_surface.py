import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from Functions.compute_cost import compute_cost

def animate_gradient_descent(X, y, theta_history):
    # Gerar valores de theta
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Calcular o custo para cada combinação de theta
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t)
    
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

    # Criar figura
    fig = plt.figure(figsize=(14, 6))

    # Subplot 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap='viridis', alpha=0.8)
    ax1.set_xlabel(r'$\theta_0$')
    ax1.set_ylabel(r'$\theta_1$')
    ax1.set_zlabel('Custo J(θ)')
    ax1.set_title('Função de Custo em 3D')

    # Subplot Contorno
    ax2 = fig.add_subplot(1, 2, 2)
    CS = ax2.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 20), cmap='viridis')
    ax2.set_xlabel(r'$\theta_0$')
    ax2.set_ylabel(r'$\theta_1$')
    ax2.set_title('Contornos da Função de Custo')
    ax2.grid(True)

    # Inicializar ponto
    point_3d, = ax1.plot([], [], [], 'ro', markersize=5)
    path_3d, = ax1.plot([], [], [], 'r-', linewidth=1)

    point_2d, = ax2.plot([], [], 'ro', markersize=5)
    path_2d, = ax2.plot([], [], 'r-', linewidth=1)

    # Atualizar animação
    def update(i):
        thetas = np.array(theta_history[:i+1])
        point_3d.set_data(thetas[-1, 0], thetas[-1, 1])
        point_3d.set_3d_properties(compute_cost(X, y, thetas[-1]))
        path_3d.set_data(thetas[:, 0], thetas[:, 1])
        path_3d.set_3d_properties([compute_cost(X, y, t) for t in thetas])

        point_2d.set_data(thetas[-1, 0], thetas[-1, 1])
        path_2d.set_data(thetas[:, 0], thetas[:, 1])
        return point_3d, path_3d, point_2d, path_2d

    ani = animation.FuncAnimation(fig, update, frames=len(theta_history), interval=100, blit=True)

    # Salvar animação (opcional)
    ani.save('Figures/animacao_descida_gradiente.gif', writer='pillow')

    plt.show()