# cargamos librerías necesarias
import numpy  as np  
import pandas as pd

import matplotlib.pyplot as plt # para dibujar

house_data = pd.read_csv("./data/kc_house_data.csv") # cargamos fichero

house_data.boxplot(by='waterfront',column = 'price')
plt.show()

