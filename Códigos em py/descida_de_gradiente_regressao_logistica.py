import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. SIMULAÇÃO DE DADOS BIOLÓGICOS
# Vamos imaginar 100 amostra com 2 biomarcadores (Expressão do Gene A e Gene B)
np.random.seed(42)
x = np.random.randn(100, 2)
# Rótulos: Se a soma da expresão for positiva, a célula é "Saudável" (1), senão "Doente" (2)
y = (x[:, 0] + x[:, 1 ] > 0).astype(int)

# 2. DIVISÃO DOS DADOS (Treino e Teste)
# Importante para o GitHub: Mostra que sabemos avaliar o modelo!!
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. ESCALONAMENTO (Feature Scaling)
# Essencial para convergência rápida
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. TREINANDO O MODELO
# Aqui a Descida de Gradiente é executada por baixo dos panos
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# 5. PREDIÇÕES E AVALIAÇÃO
y_pred = model.predict(x_test_scaled)
probabilidade = model.predict_proba(x_test_scaled)[:, 1]

# 6. VISUALIZAÇÃO DA FRONTEIRA DE DECISÃO
def plot_decision_boundary(x, y, model, title):
    h = .02 # tamanho do passo na malha
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel("Expressão Gene A")
    plt.ylabel("Expressão Gene B")

plt.figure(figsize=(10, 6))
plot_decision_boundary(X_train_scaled, y_train, model, "Limite de Decisão - Treino")
plt.show()

# 7. RELATÓRIO FINAL (Ótimo para o README do GitHub)
print("--- RELATÓRIO DE PERFORMANCE ---")
print(classification_report(y_test, y_pred))
print(f"Pesos aprendidos (w): {model.coef_}")
print(f"Viés aprendido (b): {model.intercept_}")