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