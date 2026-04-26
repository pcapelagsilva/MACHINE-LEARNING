# NESTE LABORATÓRIO IREMOS AUTOMATIZAR O PROCESSO DE OTIMIZAÇÃO DE w E b UTILIZANDO O GRADIENTE DESCENDENTE #

import math, copy 
import numpy as np # Biblioteca popular para computação científica
import matplotlib as plt # Biblioteca popular para plotagem de dados
# plt.style.use('./deeplearning.mplstyle')
'''from regressão_linear_univariada import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients''' # Rotinas de plotagem no arquivo 'lab_utils_uni.py' no diretório local

'''Carregar nosso conjunto de dados'''
x_train = np.array([1.0, 2.0]) # características (features)
y_train = np.array([300.0, 500.0]) # valor alvo (target)

''' Função para calcular o custo '''
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

# Para implementarmos o Gradiente Descendente nós precisaremos de três funções:
    # 1. compute_gradient --> implementa as derivadas;
    # 2. compute_cost --> implementa a função de custo;
    # 3. gradient_descent --> utiliza as duas funções acima simultâneamente.

def compute_gradient(x, y, w, b):
    """
    Calcula o gradiente para regressão linear 
    Args:
      x (ndarray (m,)): Dados, m exemplos 
      y (ndarray (m,)): valores alvo
      w,b (escalar)   : parâmetros do modelo  
    Returns
      dj_dw (escalar): O gradiente do custo em relação ao parâmetro w
      dj_db (escalar): O gradiente do custo em relação ao parâmetro b     
     """
    
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db = dj_db_i
        dj_dw = dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

'''plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()'''

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Executa o gradiente descendente para ajustar w,b.
    
    Args:
      x (ndarray (m,))  : Dados
      y (ndarray (m,))  : valores alvo
      w_in,b_in (escalar): valores iniciais
      alpha (float):     Taxa de aprendizado
      num_iters (int):   número de iterações
      cost_function:     função de custo
      gradient_function: função de gradiente
      
    Returns:
      w, b (escalar): Valores atualizados
      J_history (lista): Histórico de custos
      p_history (lista): Histórico de parâmetros [w,b] 
    """
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i<100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w,b])
        
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteração {i:4}: Custo {J_history[-1]:0.2e} ", 
            f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
            f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history

# Inicializar parâmetros
w_init = 0
b_init = 0
# Configurações do gradiente descendente
iterations = 10000
tmp_alpha = 1.0e-2
# Rodar gradiente descendete
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) encontrados: ({w_final:8.4f}, {b_final:8.4f})")

import matplotlib.pyplot as plt
plt.scatter(x_train, y_train, marker='x', c='r', label='Valores Reais')

previsao = w_final * x_train + b_final
plt.plot(x_train, previsao, c='b', label='Nossa Previsão')

plt.title("Preços de Casas")
plt.ylabel('Preço (em milhares de dólares)')
plt.xlabel('Tamanho (1000 sqft)')
plt.legend()
plt.show()

# VAMOS TESTAR!!!
print(f"Previsão casa 1000 sqft: {w_final*1.0 + b_final:0.1f} mil dólares")
print(f"Previsão casa 1200 sqft: {w_final*1.2 + b_final:0.1f} mil dólares")
print(f"Previsão casa 2000 sqft: {w_final*2.0 + b_final:0.1f} mil dólares")

# =-=-==-=-=-=-=-=-==- VAMOS VER OS 3 TIPOS DE GRÁFICOS TRABALHANDO -=-=-=-=-=-=-=-=-=-=-=-=-

p_history_np = np.array(p_hist) # Converte a trajetória para array NumPy

# --- Preparação da malha (grid) para os custos (Contorno e 3D) ---
# Define faixas de w e b para calcular a "bacia" de custo
# Usamos w de 0 a 400 e b de 0 a 200 para englobar w=200, b=100
w_range = np.linspace(0, 400, 100)
b_range = np.linspace(0, 200, 100)
W_grid, B_grid = np.meshgrid(w_range, b_range) # Cria a malha 2D

# Calcula o custo para cada combinação de w e b na malha
Z_cost = np.array([compute_cost(x_train, y_train, w, b) for w, b in zip(np.ravel(W_grid), np.ravel(B_grid))])
Z_cost = Z_cost.reshape(W_grid.shape) # Reajusta a forma para a malha

print("\nGerando gráficos...")

# Define a estrutura com 1 linha e 3 colunas
fig = plt.figure(figsize=(20, 6))

# --- GRÁFICO 1: Regressão Linear (Reta Ajustada) ---
ax1 = fig.add_subplot(131)
ax1.scatter(x_train, y_train, marker='x', c='r', s=100, label='Dados Reais') # s=tamanho do ponto
ax1.plot(x_train, w_final * x_train + b_final, c='b', linewidth=3, label='Reta de Previsão')
ax1.set_title("1. Melhor Ajuste Linear", fontsize=14)
ax1.set_xlabel("Tamanho (1000 sqft)", fontsize=12)
ax1.set_ylabel("Preço (1000s de dólares)", fontsize=12)
ax1.set_xticks([1.0, 2.0]) # Força mostrar os tamanhos dos dados
ax1.legend()
ax1.grid(True)

# --- GRÁFICO 2: Contorno da Função de Custo J(w,b) ---
ax2 = fig.add_subplot(132)
# Desenha as linhas de contorno coloridas e com rótulos de custo
contour = ax2.contour(W_grid, B_grid, Z_cost, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8) # Rótulos de custo nas linhas
# Plota a trajetória que o seu algoritmo percorreu (linha branca)
ax2.plot(p_history_np[:,0], p_history_np[:,1], c='white', linewidth=1, label='Trajetória do Gradiente')
# Marca o ponto final (objetivo) com uma estrela vermelha
ax2.scatter(w_final, b_final, c='red', marker='*', s=200, label='Mínimo Global')
ax2.set_title("2. Contorno de Custo J(w,b)", fontsize=14)
ax2.set_xlabel("w (Peso)", fontsize=12)
ax2.set_ylabel("b (Bias)", fontsize=12)
ax2.legend(loc='lower left')
ax2.grid(True)

# --- GRÁFICO 3: Superfície 3D da Função de Custo J(w,b) ---
ax3 = fig.add_subplot(133, projection='3d')
# Plota a superfície (tigela)
surface = ax3.plot_surface(W_grid, B_grid, Z_cost, cmap='viridis', edgecolor='none', alpha=0.8)
# Plota a trajetória descendo o poço (linha preta)
# Adicionamos J_hist na coordenada Z para a linha seguir a bacia
# Como J_hist tem mais pontos que p_history_np no início, usamos J_hist[1:]
# para alinhar se necessário. Aqui usamos o histórico completo salvo.
# Pegamos o custo de cada ponto da trajetória para a coordenada Z
cost_path = np.array([compute_cost(x_train, y_train, w, b) for w, b in p_hist])
ax3.plot(p_history_np[:,0], p_history_np[:,1], cost_path, c='black', linewidth=1, label='Descida')
ax3.set_title("3. Bacia de Custo 3D", fontsize=14)
ax3.set_xlabel("w", fontsize=12)
ax3.set_ylabel("b", fontsize=12)
ax3.set_zlabel("Custo J", fontsize=12)
ax3.view_init(elev=20, azim=130) # Ajusta o ângulo de visão inicial
# Adiciona uma barra de cores
fig.colorbar(surface, ax=ax3, shrink=0.5, aspect=10)

plt.tight_layout() # Organiza os gráficos para não sobrepor
print("Janela de gráficos abrindo no WSL...")
plt.show() # Abre a janela única com os 3 gráficos
