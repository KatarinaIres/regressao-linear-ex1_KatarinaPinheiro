# regressao-linear-ex1.py

import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.warm_up_exercises import warm_up_exercise
from Functions.plot_data import plot_data
from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent

# 1. Garantir que a pasta Figures existe
os.makedirs('Figures', exist_ok=True)

# 2. Exercício de aquecimento
print("🔹 Matriz identidade 5x5:")
print(warm_up_exercise())
print("\n==============================\n")

# 3. Carregar os dados
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Data', 'ex1data1.txt')

print(f"🔹 Carregando dados de treinamento de {file_path}...")
data = np.loadtxt(file_path, delimiter=',')
X_original = data[:, 0]   # Salva X original
y = data[:, 1]
m = len(y)

# 4. Visualizar os dados
plt.figure()
plot_data(X_original, y, color='blue', marker='o', label='Dados de treinamento')
plt.title('Dados de Treinamento')
plt.xlabel('População da cidade')
plt.ylabel('Lucro')
plt.legend()
plt.grid(True)
plt.savefig('Figures/dados_treinamento.png')
plt.close()

# 5. Preparar os dados para treinamento
X = X_original.reshape((m, 1))
X = np.concatenate([np.ones((m, 1)), X], axis=1)  # Adiciona coluna de 1's
theta = np.zeros(2)  # Inicializa theta

# 6. Definir parâmetros do gradiente
iterations = 1500
alpha = 0.01

# 7. Custo inicial
initial_cost = compute_cost(X, y, theta)
print(f"🔹 Custo inicial com theta zeros: {initial_cost:.4f}")

# 8. Rodar descida do gradiente
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
print(f"🔹 Theta encontrado: {theta}")
print("\n==============================\n")

# 9. Plotar a linha de regressão
plt.figure()
plot_data(X_original, y, color='blue', marker='o', label='Dados de treinamento')
plt.plot(X_original, np.dot(X, theta), color='red', label='Regressão Linear')
plt.title('Ajuste Linear aos Dados')
plt.xlabel('População da cidade')
plt.ylabel('Lucro')
plt.legend()
plt.grid(True)
plt.savefig('Figures/ajuste_regressao.png')
plt.close()

# 10. Plotar o custo vs. iterações
plt.figure()
plt.plot(range(1, iterations + 1), J_history, color='purple')
plt.title('Convergência do Custo')
plt.xlabel('Número de Iterações')
plt.ylabel('Custo J(θ)')
plt.grid(True)
plt.savefig('Figures/convergencia_custo.png')
plt.close()