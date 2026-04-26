import numpy as np

def calcular_custo_logistico(x, y, w, b):
    """ Calcula o custo j(w,b) usando a versão simplificada (log-loss) """
    m = x.shape[0] # Número de exemplos

    # 1. Calcula a previsão de f(x) para todos os exemplos (Vetorizado)
    # z = w * x + b
    z = np.dot(x, w) + b
    # f(x) = g(z) -> Função Sigmoide
    f_x = 1 / (1 + np.exp(-z))

    # 2. Calcula a Perda (loss) usando a fórmula simplificada de uma linha:
    # l = -y*log(f) - (1-y)*log(1-f)
    # np.log() calcula o logaritmo natural
    perda = -y * np.log(f_x) - (1 - y) * np.log(1 - f_x)

    # 3. O Custo j é a média da perda
    custo_total = (1 / m) * np.sum(perda)

    return custo_total


# --- TESTE ---
# Dados os exemplos: 2 recursos (tamanho tumor, idade)
x_treino = np.array([[0.5, 1.5], [1.1, 1.9], [3.0, 3.2], [4.5, 5.1]])
y_treino = np.array([0, 0, 1, 1]) # Rótulos reais

# Parâmetros iniciais (w e b)
w_inicial = np.array([0.2, 0.2])
b_inicial = -1.5

custo = calcular_custo_logistico(x_treino, y_treino, w_inicial, b_inicial)

print(f"Custo j(w,b) com parâmetros iniciais: {custo:.4f}")
