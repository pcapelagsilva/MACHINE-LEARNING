import numpy as np

# Parâmetros de exemplo
alpha = 0.01 # Taxa de aprendizado
lambd = 1.0 # Parâmetro de regularização (lambda)
m = 50 # Tamanho do conjunto de treinamento
w_j = 10.0 # Valor inicial do peso

print(f"Valor inicial de w_j: {w_j}")

# 1. Calculando o fator de encolhimento
fator_encolhimento = 1 - (alpha*(lambd/m))
print(f"Fator de encolhimento por iteração: {fator_encolhimento:.4f}")

# 2. Simulando 10 iterações de encolhimento (sem contar o erro ainda)
for i in range(1, 11):
    w_j = w_j * fator_encolhimento
    print(f"Iteração {i}: w_j encolheu para {w_j:.4f}")

print("\nConclusão: Mesmo sem calcular o erro dos dados,")
print("a regularização já esta 'puxando' o peso para baixo")