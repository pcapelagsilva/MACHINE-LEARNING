import numpy as np

# 1. DADOS ORIGINAIS: [Largura do terreno, Profundidade do terreno]
x_original = np.array([
    [20, 50], # Casa 1
    [15, 40], # Casa 2
    [25, 60] # Casa 3
])

# 2. ENGENHARIA DE ATRIBUTOS: Criando a "Área"
''' Multiplicamos a primeira coluna (largura) pela segunda (profundidade)'''
# O [:, 0] pega todas as linhas da coluna 0
# O [:, 1] pega todas as linhas da coluna 1
area = x_original[:, 0] * x_original[:, 1]

# Redimensionamos "area" para ser uma coluna (3 linhas, 1 coluna)
area = area.reshape(-1, 1)

# 3. UNINDO OS ATRIBUTOS: [Largura, Profundidade, Área]
# np.hstack empilha os arrays horizontalmente
x_treinamento = np.hstack((x_original, area))

print("Matriz Original (Largura, Profundidade):")
print(x_original)
print("\nNovo Atributo (Área):")
print(area)
print("\nMatriz Final para o Modelo (Largura, Profundidade, Área):")
print(x_treinamento)