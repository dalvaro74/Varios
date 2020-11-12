# Grafica de y = X al cuadrado
import matplotlib.pyplot as plt
x=[0.1*i for i in range(-50,51)] # x=[-5, -4,9, ….., 5.1]
y=[x_i**2 for x_i in x] # y=x2
plt.xlabel('x') # etiqueta X
plt.ylabel('y$^2$') # Entiende Latex
plt.plot(x,y)
plt.show()