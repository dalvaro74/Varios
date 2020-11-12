from matplotlib import pyplot as plt
import numpy as np

def gaussian(x, a, b, c):
     return a*np.exp(-np.power((x - b)/c, 2.)/2)

x_values = np.linspace(-12, 12, 120)
for a, b, c in [(3, 0, 1), (3, 0, 2), (3, 0, 3)]:
    plt.xlim(-15, 15)
    plt.plot(x_values, gaussian(x_values, a, b, c))

plt.show()