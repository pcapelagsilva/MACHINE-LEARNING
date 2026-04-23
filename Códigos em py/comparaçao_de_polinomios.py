import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. GERANDO DADOS (Uma curva senoidal com um pouco de ruído)
np.random.seed(0)
n_sample = 30
x = np.sort(np.random.rand(n_sample))
y = np.cos(1.5 * np.pi * x) + np.random.randn(n_sample) * 0.1

# 2. GRAUS DOS POLINÔMIIOS PARA COMPARAR
# Grau 1 = Underfitting | Grau 4 = Just Right | Grau 15 = Overfitting
degrees = [1, 4, 15]

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)

    # Criando o modelo polinomial
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("poly", polynomial_features), ("linear", linear_regression)])

    pipeline.fit(x[:, np.newaxis], y)

    # Plotando os resultados
    x_test = np.linspace(0, 1, 100)
    plt.plot(x_test, pipeline.predict(x_test[:, np.newaxis]), label="Modelo")
    plt.scatter(x, y, edgecolors='b', s=20, label="Dados de Treino")

    plt.xlabel("Tamanho da Casa")
    plt.ylabel("Preço")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(f"Grau {degrees[i]}\n"+
              ("Underfitting" if degrees[i]==1 else "Overfitting" if degrees[i]==15 else "Just Right"))
    
plt.tight_layout()
plt.show()