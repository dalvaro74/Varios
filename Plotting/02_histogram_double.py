import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("./data/breast_cancer.csv", sep=",", decimal=".")
data.drop(['id','Unnamed: 32'], inplace=True, axis=1)

# Convertimos ahora diagnosis en una variable numérica. 
data['diagnosis'] = np.where(data['diagnosis'] == 'M',1,0)

# Pintamos histogramas para cada clase
plt.figure(figsize=(20,20))

idx_0 =  data['diagnosis'] == 0
idx_1 =  data['diagnosis'] == 1
#idx_0
#data['radius_mean']
#data.loc[idx_0,'radius_mean']

for i,feature in enumerate(data.columns.drop(['diagnosis'])):
    plt.subplot(6,5,i+1)   
    plt.hist(data.loc[idx_0,feature],density=1, alpha=0.75,label='y=0')
    plt.hist(data.loc[idx_1,feature],density=1, facecolor='red', alpha=0.75,label='y=1')
    plt.legend()
    plt.title(feature)

plt.show()


