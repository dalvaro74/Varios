from matplotlib import pyplot as plt
import numpy as np

def gaussian(x, mu, sig):
    #return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

#x_values = np.linspace(-8, 8, 100)
x_values = np.arange(-8, 8, 0.1)
for mu, sig in [(0, 0.1), (0, 2), (0, 3)]:
    plt.xlim(-10, 15)
    plt.plot(x_values, gaussian(x_values, mu, sig))


plt.show()