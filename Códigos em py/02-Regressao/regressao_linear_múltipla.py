import numpy as np
import copy, math
import matplotlib.pyplot as plt

# --- 1. DEFINIÇÃO DO MODELO (DADOS) ---
# x_train = Matriz com os recursos (ex: Tamanho, Quartos, Andares e Idade)
# y_train = Saida (ex: Preço da casa em milhares de dólares)
x_train = np.array([
    [2104, 5, 1, 45], 
    [1416, 3, 2, 40], [852, 2, 1, 35]
    ])

y_train = np.array([460, 232, 178])

# Parâmetros iniciais (Vetor w e Escalar b)
# Para valores com b diferente de 0, basta zerar todos os valores de b_init e w_init
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# --- 2. CÁLCULO DA PREVISÃO (VETORIZADO) ---
def predict(x, w, b):
    '''
    Calcula f_wb = w * x + b usando o produto escalar (dot product).
    '''
    p = np.dot(x, w) + b
    return p

# --- 3. CÁLCULO DO CUSTO ---
def compute_cost(X, y, w, b):
    '''
    Calcula o custo J(w,b) para múltiplas variáveis.
    '''
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2*m)
    return cost

# --- 4. CÁCULO DO GRADIENTE ---
def compute_gradient(X, y, w, b):
    '''
    Calcula as derivadas parciais dj_dw (vetor) e dj_db (escalar).
    '''
    m, n = X.shape    # m: exemplos, n: recursos
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

# --- 5. DESCIDA DE GRADIENTE ---
def gradient_descent(X, y, w_in, b_in, cost_fn, grad_fn, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = grad_fn(X, y, w, b)

        # Atualização simultânea de w (vetor) e b (escalar)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000: J_history.append(cost_fn(X, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4d}: Custo {J_history[-1]:8.2f}")
    
    return w, b, J_history

# --- EXECUÇÃO ---
iterations = 1000
alpha = 5.0e-7 # Learning rate pequeno para evitar divergência sem normalização
w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)

print(f"\nResultado Final: b = {b_final:0.2f}, w = {w_final}")
for i in range(3):
    print(f"Previsão {i}: {np.dot(x_train[i], w_final) + b_final:0.2f}, Alvo: {y_train[i]}")
