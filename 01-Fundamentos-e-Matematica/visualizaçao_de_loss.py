import numpy as np
import matplotlib.pyplot as plt

def visuzalizar_funcao_de_perda():
    # Gerando previsões de 0.001 a 0.999 (evitando 0 e 1 exatos para o log não explodir)
    f_x = np.linspace(0.001, 0.999, 100)

    # 1. Perda quando o rótulo verdadeiro y = 1
    loss_y1 = -np.log(f_x) #Em computação, o logaritmo natural ($\ln$) é a base para o cálculo de entropia e informação, conceitos fundamentais na biologia e na ciência de dados.

    # 2. Perda quando o rótulo verdadeiro y = 0
    loss_y0 = -np.log(1 - f_x)

    plt.figure(figsize=(12, 5))

    # Gráfico para y = 1
    plt.subplot(1, 2, 1)
    plt.plot(f_x, loss_y1, color='purple', linewidth=3)
    plt.title("Perda se o Rótulo Real y = 1\n(Tumor Maligno)")
    plt.xlabel("Previsão do Modelo f(x)")
    plt.ylabel("Perda (Erro)")
    plt.grid(alpha=0.3)
    plt.annotate('Erro Mínimo', xy=(1, 0), xytext=(0.6, 2), arrowprops=dict(facecolor='black', shrink=0.05))

    # Gráfico para y = 0
    plt.subplot(1, 2, 2)
    plt.plot(f_x, loss_y0, color='blue', linewidth=3)
    plt.title("Perda se o Rótulo Real y = 0\n(Tumor Benigno)")
    plt.xlabel("Previsão do Modelo f(x)")
    plt.ylabel("Perda (Erro)")
    plt.grid(alpha=0.3)
    plt.annotate('Erro Mínimo', xy=(0, 0), xytext=(0.2, 2), arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visuzalizar_funcao_de_perda()
