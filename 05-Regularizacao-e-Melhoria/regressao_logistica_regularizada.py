import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Criando dados sintéticos circulares (comuns em bioinfo)
np.random.seed(0)
x = np.random.randn(100, 2)
y = (x[:, 0]**2 + x[:, 1]**2 > 1).astype(int)

# Polinômio de alta ordem para forçar a complexidade
grau = 6

plt.figure(figsize=(12, 5))

# 2. Modelo com POUCA Regularização (C grande no sklearn significa lambda pequeno)
# No Scikit-Learn, o parâmetro 'C' é o inverso de lambda (C = 1/lambda)
model_overfit = make_pipeline(PolynomialFeatures(grau), LogisticRegression(C=1e5))
model_overfit.fit(x, y)

# 3. Modelo com MUITA Regularização (C pequeno = Lambda grande)
model_regularized = make_pipeline(PolynomialFeatures(grau), LogisticRegression(C=0.1))
model_regularized.fit(x, y)

# Função para plotar o limite de decisão
def plot_boundary(model, title, subplot):
    plt.subplot(1, 2, subplot)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
    plt.title(title)

plot_boundary(model_overfit, "Overfitting (Lambda ≈ 0)", 1)
plot_boundary(model_regularized, "Regularizado (Lambda Ideal)", 2)

plt.tight_layout()
plt.show()
