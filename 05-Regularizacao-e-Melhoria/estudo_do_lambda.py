import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Dados simulados
np.random.seed(42)
X = np.sort(5 * np.random.rand(20, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# Vamos usar um polinômio de grau 10 (muito complexo)
grau = 10
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='black', label='Dados Reais (com ruído)')

# 2. Testando 3 valores de Lambda (alpha no sklearn)
lambdas = [0, 0.01, 1000] # Lambda 0, Médio e Gigante
cores = ['red', 'green', 'blue']
estilos = ['--', '-', '-.']

for l, cor, estilo in zip(lambdas, cores, estilos):
    # Ridge é a Regressão Linear com Regularização L2
    model = make_pipeline(PolynomialFeatures(grau), Ridge(alpha=l))
    model.fit(X, y)
    
    label = f'Lambda = {l} ' + ('(Overfitting)' if l==0 else '(Underfitting)' if l==1000 else '(Ideal)')
    plt.plot(X_plot, model.predict(X_plot), color=cor, linestyle=estilo, label=label, linewidth=2)

plt.title("O Impacto do Parâmetro de Regularização (Lambda)")
plt.ylim(-2, 2)
plt.legend()
plt.show()