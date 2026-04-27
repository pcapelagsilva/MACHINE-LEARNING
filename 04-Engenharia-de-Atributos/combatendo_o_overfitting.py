import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Dados simulados (poucos dados e muita curva = receita para overfitting)
np.random.seed(42)
x = np.array([0, 1, 2, 3, 4, 5])[:, np.newaxis]
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])

# Criamos um polinômio de grau 5 (complexo demais para 6 pontos)
grau = 5

# 2. Modelo SEM Regularização (Overfitting)
model_overfit = make_pipeline(PolynomialFeatures(grau), LinearRegression())
model_overfit.fit(x, y)

# 3. Modelo COM Regularização (Ridge / L2)
# O parâmetro 'alpha' controla a força da regularização (quanto maior, menores os pesos)
model_regularized = make_pipeline(PolynomialFeatures(grau), Ridge(alpha=1.0))
model_regularized.fit(x, y)

# Visualização
x_plot = np.linspace(0, 5, 100)[:, np.newaxis]
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Dados de Treino')
plt.plot(x_plot, model_overfit.predict(x_plot), '--', label='Sem Regularização (Overfitting)')
plt.plot(x_plot, model_regularized.predict(x_plot), label='Com Regularização (Suave)', linewidth=3)

plt.title("O Poder da Regularização")
plt.legend()
plt.ylim(-2, 2)
plt.show()

# Comparando os Pesos (w)
print("Pesos sem regularização (são muito altos):")
print(model_overfit.named_steps['linearregression'].coef_)
print("\nPesos COM regularização (são muito menores):")
print(model_regularized.named_steps['ridge'].coef_)