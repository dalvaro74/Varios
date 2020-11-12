import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("./data/breast_cancer.csv", sep=",", decimal=".")
data.drop(['id','Unnamed: 32'], inplace=True, axis=1)
type(data.columns)

features = data.columns.drop(['diagnosis'])
len(features)

# Asignamos valores aleatorios a esas features entre 0 y 100
#np.random.seed(55)
feat_values = np.random.randint(0,101, len(features))

# Ordenamos los indices en orden descendente
indices = np.argsort(feat_values)[::-1]

# Si quisieramos ordenar los valores en orden descendente...
values_sort = np.sort(feat_values)[::-1]
 
plt.figure(figsize=(10,10))
plt.barh(range(len(features)),feat_values[indices])
plt.yticks(range(len(features)),features[indices])
plt.show()