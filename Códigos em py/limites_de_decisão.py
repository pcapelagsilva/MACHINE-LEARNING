import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary():
    # Criando uma grade de pontos para o fundo do gráfico
    x_range = np.linspace(-1, 5, 100)
    y_range = np.linspace(-1, 5, 100)
    x1, x2 = np.meshgrid(x_range, y_range) # Cria uma grade de coordenadas

    plt.figure(figsize=(12, 5))

    # --- EXEMPLO 1: LIMITE LINEAR ---
    # Parâmetros: w1=1, w2=1, b=-3
    # z = x1 + x2 - 3
    plt.subplot(1, 2, 1)
    z_linear = x1 + x2 - 3

    # Desenha a linha roxa onde z = 0
    plt.contour(x1, x2, z_linear, levels=[0], colors='purple', linewidths=3)
    # Pinta as áreas de decisão
    plt.contourf(x1, x2, z_linear, levels=[-10, 0, 10], alpha=0.2, colors=['blue', 'red'])
    
    plt.title("Limite de Decisão Linear\n(z = x1 + x2 - 3)")
    plt.xlabel("Recurso x1")
    plt.ylabel("Recurso x2")
    plt.grid(alpha=0.3)

    # --- EXEMPLO 2: LIMITE CIRCULAR (POLINOMIAL) ---
    # Parâmetr0s: w1=1, w2=1, b=-1 (para temros quadráticos)
    # z = x1^2 + x2^2 - 1
    plt.subplot(1, 2, 2)
    # Centralizando o círculo em (2,2) para melhor visualização
    z_circ = (x1-2)**2 + (x2-2)**2 -1

    # Desenha o círculo onde z = 0
    plt.contour(x1, x2, z_circ, levels=[0], colors='purple', linewidths=3)
    # Pinta a área interna (azul < 0) e externa (vermelha > 0)
    plt.contourf(x1, x2, z_circ, levels=[-10, 0, 10], alpha=0.2, colors=['blue', 'red'])

    plt.title("Limite de Decisão Circular\n(z = x1² + x2² - 1)")
    plt.xlabel("Recurso x1")
    plt.ylabel("Recurso x2")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_decision_boundary()