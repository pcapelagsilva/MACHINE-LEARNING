import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. DADOS DE EXEMPLO (Simulando crescimento biológico)
''' x = Horas passadas | y = Quantidade de bactérias'''
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2, 5, 9, 18, 33, 60, 110, 200])

# 2. ENGENHARIA DE ATRIBUTOS (Transformando para Polinomial)
''' Vamos criar x, x^2 e x^3 para capturar a curva de crescimento '''
grau = 3
poly_features = PolynomialFeatures(degree=grau, include_bias=False)
x_poly = poly_features.fit_transform(x)

# 3. ESCALONAMENTO (Z-Score / StandardScaler)
''' Essencial para que o x^3 não "atropele" o x no Gradiente Descendente '''
scaler = StandardScaler()
x_poly_scaled = scaler.fit_transform(x_poly)

# 4. TREINANDO O MODELO
''' O Scikit-learn resolve o "w" e o "b" de forma otimizada '''
model = LinearRegression()
model.fit(x_poly_scaled, y)

# 5. PREDIÇÃO PARA O GRÁFICO (Criando uma linha suave)
x_suave = np.linspace(1, 8, 100).reshape(-1, 1) # 100 pontos entre 1 e 8
x_suave_poly = poly_features.transform(x_suave) # Transforma para polinomial
x_suave_scaled = scaler.transform(x_suave_poly) # Usa APENAS transform, não fit_transform
y_pred = model.predict(x_suave_scaled) # Faz a previsão

# 6. VISUALIZAÇÃO EM GRÁFICO DOS RESULTADOS
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Dados Observados', zorder=5)
plt.plot(x_suave, y_pred, color='blue', linewidth=2, label=f'Modelo Polinomial (Grau {grau})')

plt.title("Regressão Polinomial: Crescimento de Cultura", fontsize=14)
plt.xlabel("Tempo (Horas)", fontsize=12)
plt.ylabel("População", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Exibindo os parâmetros aprendidos
print(f"--- PARÂMETROS DO MODELO ---")
print(f"Intercepto (b): {model.intercept_:.2f}")
print(f"Pesos (w1, w2, w3); {model.coef_}")

# Fazendo a previsão nos dados originais para comparar
y_treino_pred = model.predict(x_poly_scaled)

r2 = r2_score(y, y_treino_pred)
mse = mean_squared_error(y, y_treino_pred)

print(f"R² Score: {r2:.4f}") # Quanto mais próximo de 1.0, melhor
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")