import numpy as np

''' Dados Originais: [Tamanho (sqft), Quartos] '''
x_train = np.array([
    [2000, 5],
    [1600, 3],
    [1200, 2],
    [300, 1]
])

# --- 1. DIVISÃO PELO MÁXIMO ---
''' Divide cada coluna pelo seu valor máximo'''
x_max = np.max(x_train, axis=0)
x_scaled_max = x_train/x_max

# --- 2. NORMALIZAÇÃO MÉDIA ---
''' (Valor - Média) / (Máximo - Mínimo) '''
mu = np.mean(x_train, axis=0)
range_val = np.max(x_train, axis=0) - np.min(x_train, axis=0)
x_normalized_mean = (x_train - mu)/range_val

# --- 3. NORMALIZAÇÃO Z-SCORE (PADRONIZAÇÃO) ---
''' (Valor - Média) / Desvio Padrão '''
sigma = np.std(x_train, axis=0)
x_zscore = (x_train - mu)/sigma

# --- EXIBIÇÃO DOS RESULTADOS ---
print("Dados Originais:\n", x_train)
print("\n1. Divisão pelo Máximo (0 a 1):\n", x_scaled_max)
print("\n2. Normalização Média (-1 a 1 aprox.):\n", x_normalized_mean)
print("\n3. Z-score (Centralizado em 0):\n", x_zscore)
