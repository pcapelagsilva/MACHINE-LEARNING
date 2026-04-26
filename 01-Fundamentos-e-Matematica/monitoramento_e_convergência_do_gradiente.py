import matplotlib.pyplot as plt

# Exemplo de valor para j_history (Custo por iteração)
j_history = [
    79300.0, 45000.0, 21000.0, 8500.0, 3200.0, 1100.0, 450.0, 180.0, 75.0, 32.0, 
    14.5, 6.8, 3.2, 1.5, 0.75, 0.40, 0.22, 0.12, 0.08, 0.05, 
    0.04, 0.035, 0.032, 0.031, 0.03, 0.03, 0.03 # Neste último valor ele estabilizou (convergiu)
]

# Exemplo rápido de como plotar a curva que você estudou
plt.plot(j_history)
plt.title("Curva de Aprendizado")
plt.xlabel("Iterações")
plt.ylabel("Custo j")
plt.grid(True)
plt.show()
