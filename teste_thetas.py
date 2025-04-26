# teste_thetas.py

import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent

# Carregar os dados
data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

# Normalizar a feature X
X_mean = np.mean(X)
X_std = np.std(X)
X_norm = (X - X_mean) / X_std

# Preparar X com a coluna de 1s
X_norm = X_norm.reshape((m, 1))
X_norm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# Definir alpha e número de iterações
alpha = 0.01
iterations = 400

# Diferentes inicializações de theta (corrigido para float)
theta_inits = [
    np.array([0, 0], dtype=float),
    np.array([1, 1], dtype=float),
    np.array([-1, -1], dtype=float),
    np.array([2, 0], dtype=float),
    np.array([0, 2], dtype=float),
    np.array([-2, -1], dtype=float)
]

# Garantir que o diretório Figures/ existe
os.makedirs('Figures', exist_ok=True)

# Criar figura
plt.figure()

for i, theta_init in enumerate(theta_inits):
    theta = theta_init.copy()
    theta, J_history = gradient_descent(X_norm, y, theta, alpha, iterations)
    plt.plot(range(1, iterations + 1), J_history, label=f'Theta inicial {i+1}: {theta_init}')

plt.title('Comparação de inicializações de θ')
plt.xlabel('Número de Iterações')
plt.ylabel('Custo J(θ)')
plt.legend()
plt.grid(True)

# Salvar a figura
fig_path = 'Figures/comparacao_thetas.png'
plt.savefig(fig_path)
plt.close()

print(f'Figura salva em {fig_path}')
