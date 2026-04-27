# ⚖️ Regularização e Otimização de Modelos

Este módulo é dedicado ao estudo de técnicas avançadas para combater o **Overfitting** (Sobreajuste) e garantir que os algoritmos de Machine Learning possuam uma alta capacidade de **Generalização**. 

Em contextos de Bioinformática, onde lidamos com um alto número de variáveis (como múltiplos biomarcadores) e amostras limitadas, a regularização é essencial para evitar diagnósticos errôneos baseados em ruídos estatísticos.

## 🧠 Conceitos Fundamentais

### 1. O Problema: Overfitting (Alta Variância)
Ocorre quando o modelo se torna tão complexo que "decora" os dados de treinamento, perdendo a capacidade de prever novos casos. No gráfico, isso se manifesta como uma curva extremamente sinuosa e errática.

### 2. A Solução: Regularização L2 (Ridge)
A técnica adiciona um termo de penalidade à **Função de Custo** original. Isso força o algoritmo a manter os parâmetros ($w_j$) pequenos, resultando em uma curva mais suave e robusta.

**Nova Função de Custo:**
$$J(\vec{w},b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

### 3. O Papel do Lambda ($\lambda$)
O parâmetro $\lambda$ controla o equilíbrio entre dois objetivos:
- **$\lambda = 0$:** Foca apenas em ajustar os dados (Risco de Overfitting).
- **$\lambda$ Grande:** Foca em manter os pesos pequenos (Risco de Underfitting).
- **$\lambda$ Ideal:** O equilíbrio perfeito para a melhor generalização.

---

## 💻 Implementações neste Módulo

### [Combatendo o Overfitting](./combatendo_o_overfitting.py)
Script que compara visualmente um modelo polinomial de alta ordem sem proteção contra um modelo utilizando regularização. 
- **Destaque:** Observação da redução drástica nos valores dos pesos ($w$) no console após a aplicação da técnica.

### [Estudo do Impacto do Lambda](./estudo_do_lambda.py)
Uma análise comparativa que demonstra como diferentes valores de $\lambda$ transformam a fronteira de decisão, desde o ajuste excessivo até a simplificação extrema.

---

## 📊 Visualização dos Resultados

![Gráfico de Comparação de Regularização](resultado_regularizacao.jpg)
> *Legenda: Comparação entre o ajuste polinomial livre (tracejado) e o ajuste regularizado (sólido). A regularização impede que o modelo siga ruídos aleatórios nos dados.*

---

## 🛠️ Tecnologias Utilizadas
- **Python 3.x**
- **Scikit-Learn:** (Ridge Regression, PolynomialFeatures, Pipeline)
- **Matplotlib:** Para visualização das curvas de ajuste.
- **NumPy:** Processamento vetorial dos dados.