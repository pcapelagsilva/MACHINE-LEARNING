import numpy as np # Biblioteca popular de computação científica
%matplotlib widget
import matplotlib.pyplot as plt # Biblioteca popular de plotagem de dados
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl # Rotina local de plotagem no arquivo
plt.style.use('./deeplearning.mplstyle')

''' Queremos um modelo que preve preços de casas dado o tamanho da casa.
Usaremos os mesmos dois pontos do "regressão_linear_univariada.py", uma casa com 1000 sqft vendida à $300.000, e uma casa de 2000 sqft vendida por $500.000'''
#def casa1(tamanho1, preço1):
#    tamanho1 = 1000
#    preço1 = 300000
#def casa2(tamanho2, preço2):
#   tamanho2 = 2000
#    preço2 = 500000
x_train = np.array([1.0, 2.0]) # Tamanho em 1000sqft
y_train = np.array([300.0, 500.0]) # Preço em milhares de dólares

''' O termo "cost" neste exercício pode ser confuso, já que os dados tratam do "custo da casa" (preço).
Aqui, "cost" é uma medida de quão bem nosso modelo está prevendo o preço alvo da casa. 
O termo "preço" é usado para os dados das casas.
A equação para o custo da variável é "j(w,b)" onde "f(x) = w*x+b"
  - f(x^i) --> É a previsão para o exemplo i usado os parâmetros w,b
  - (f(x^i) - y^i)^2 --> É a diferença quadrada entre o valor alvo e a previsão
Essas difereças são somadas em todos os "m" exemplos e divididas por "2m" para produzir o custo, j(w,b)'''

'''O código abaixo calcula o cost percorrendo cada exemplo em um loop. Em cada iteração:
  -> f_wb (a previsão) é calculada;
  -> A diferença entre o alvo e a previsão é calculada e elevada ao quadrado;
  -> Isso é adicionado ao custo total'''
def compute_cost(x, y, w, b):
    """
    Calcula a função de custo para regressão linear.
    
    Args:
      x (ndarray (m,)): Dados, m exemplos 
      y (ndarray (m,)): valores alvo (targets)
      w,b (escalar)    : parâmetros do modelo  
    
    Returns
        total_cost (float): O custo de usar w,b como parâmetros para ajustar os pontos em x e y
    """
    # número de exemplos de treinamento
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

"""
Seu objetivo é encontrar um modelo $f_{w,b}(x) = wx + b$ que preveja com precisão os valores das casas. O custo mede quão preciso o modelo é nos dados de treinamento.
Se w e b forem selecionados de forma que as previsões correspondam exatamente aos dados reais y, o termo de diferença quadrada será zero e o custo será minimizado.
No laboratório anterior, você determinou que b=100 fornecia uma solução ideal. Vamos fixar b=100 e focar em w. 
O gráfico resultante mostrará que:
    -> O custo é minimizado quando w=200;
    -> O custo aumenta rapidamente quando w é muito grande ou muito pequeno (devido ao termo ao quadrado).
"""

"""
Como resultado vemos como o custo varia em relação a ambos w e b através do gráfico 3D ou um gráfico de contorno (contour plot).
É instrutivo visualizar um cenário com mais pontos de dados que não estão na mesma linha.
  -> Nesse caso, é impossível encontrar w e b que resultem em um custo exatamente zero (0);
  -> As linhas tracejadas do gráfico representam a porção do custo contribuida por cada exemplo individual
"""

"""
O fato da função de custo elevar o erro ao quadrado garante que a "superficie de erro" seja CONVEXA (soup bowl). Ela sempre terá um único ponto
mínimo que pode ser alcançado seguindo o gradiente em todas as dimensões.
"""

""" O gradiente seria mais ou menos assim:"""
tmp_w = w - alpha * dj_dw
tmp_b = b - alpha * dj_db
w = tmp_w
b = tmp_b

""" Ele é considerado correto pois ambos os parâmetros foram renovados simultaneamente"""