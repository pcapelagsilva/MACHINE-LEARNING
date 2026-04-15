import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# 1. Dados de exemplo: Tamanho do tumor vs Maligno (0=Não, 1=Sim)
X = np.array([0.5, 1, 1.5, 2, 5, 5.5, 6, 6.5]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# 2. Adicionando um 'Outlier' (um tumor muito grande que deslocaria a reta)
X_com_outlier = np.append(X, [[20]], axis=0)
y_com_outlier = np.append(y, [1])

# 3. Criando os modelos
modelo_linear = LinearRegression().fit(X_com_outlier, y_com_outlier)
modelo_logistico = LogisticRegression().fit(X, y)

# 4. Visualização
plt.figure(figsize=(10, 5))
plt.scatter(X_com_outlier, y_com_outlier, color='red', label='Dados (Tumores)')

# Linha da Regressão Linear
eixo_x = np.linspace(0, 22, 100).reshape(-1, 1)
previsao_linear = modelo_linear.predict(eixo_x)
plt.plot(eixo_x, previsao_linear, '--', label='Reta da Regressão Linear', color='gray')

# Curva da Regressão Logística (O que aprenderemos a seguir)
previsao_logistica = modelo_logistico.predict_proba(eixo_x)[:, 1]
plt.plot(eixo_x, previsao_logistica, color='blue', linewidth=2, label='Curva Logística (S-shape)')

# Configurações do gráfico
plt.axhline(0.5, color='black', linestyle=':', label='Limite de Decisão (0.5)')
plt.ylim(-0.2, 1.2)
plt.title("Por que a Regressão Linear falha na Classificação?")
plt.xlabel("Tamanho do Tumor")
plt.ylabel("Probabilidade / Classe")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

'''
O que observar nesse código:
    --> A Reta Cinza (Linear): Note como ela tenta alcançar o ponto distante (20) e acaba cruzando a linha de 0.5 muito mais para a direita, errando os tumores de tamanho 4 ou 5.--> A Curva Azul (Logística): Ela tem um formato de "S". Ela se mantém entre $0$ e $1$, ignorando o efeito do ponto distante e mantendo a classificação correta.
'''