# teste_alphas.py

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

# Definir valores de alpha
alphas = [0.01, 0.03, 0.1]
iterations = 400
theta_init = np.zeros(2)

# Garantir que o diretório Figures/ existe
os.makedirs('Figures', exist_ok=True)

# Criar figura
plt.figure()

for alpha in alphas:
    theta = theta_init.copy()
    theta, J_history = gradient_descent(X_norm, y, theta, alpha, iterations)
    plt.plot(range(1, iterations + 1), J_history, label=f'α = {alpha}')

plt.title('Comparação de taxas de aprendizado')
plt.xlabel('Número de Iterações')
plt.ylabel('Custo J(θ)')
plt.legend()
plt.grid(True)

# Salvar a figura
fig_path = 'Figures/comparacao_alphas.png'
plt.savefig(fig_path)