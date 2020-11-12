import math
import numpy as np 
from matplotlib import pyplot as plt
#Es lo mismo que
# from matplotlib.pyplot as plt


x = np.array(range(200))*0.1
y = np.zeros(len(x))

for i in range(len(x)):
    y[i] = np.sin(x[i])

#Creamos el grafico
plt.plot(x,y)
plt.show()

 