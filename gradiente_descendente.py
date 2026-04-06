import numpy as np

def compute_gradient(x, y, w, b):
    """
    Calcula as derivadas (gradientes) para a regressão linear.
    Args:
      x (ndarray (m,)): Dados de entrada (ex: tamanho da casa)
      y (ndarray (m,)): Valores alvo (ex: preço real)
      w,b (scalar)    : Parâmetros do modelo
    Returns
      dj_dw (scalar): O gradiente do custo em relação ao parâmetro w
      dj_db (scalar): O gradiente do custo em relação ao parâmetro b
    """
    m = x.shape[0] # Número de exemplos treinados
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        # 1. Calcula a previsão (f_wb)
        f_wb = w * x[i] + b

        # 2. Calcula o erro (previsão - valor real)
        err = f_wb - y[i]

        # 3. Soma os gradientes conforme a fórmula matemática
        # O segredo está em acumular corretamente aqui:
        dj_dw += err * x[i]
        dj_db += err
        
    return dj_dw / m, dj_db / m

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Executa o gradiente descendente para encontrar w e b otimizados.
    """
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        # Calcula os gradientes atuais
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # Atualização simultânea
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Imprime o progresso a cada 1000 iterações
        if i % 1000 == 0:
            print(f"Iteração {i}: w={w:.3f}, b={b:.3f}")
            
    return w, b

# -------------------------------------------- TESTE COM ALFA SEGURO --------------------------------------------------------
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# IMPORTANTE: Use alpha 0.01 para não travar no 150!
w_final, b_final = gradient_descent(x_train, y_train, 0, 0, 0.01, 20000)
print(f"\nRESULTADO FINAL: w = {w_final:.2f}, b = {b_final:.2f}")