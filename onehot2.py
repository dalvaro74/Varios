import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('./data/bikes.csv',sep=';', decimal='.')
data.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)

from datetime import datetime

data['dteday'] = data['dteday'].apply(lambda x: datetime.strptime(x,'%d-%m-%Y'))
data['year'] = data['dteday'].apply(lambda x: x.year - 2011)
data['month'] = data['dteday'].apply(lambda x: x.month)
data['weekday'] = data['dteday'].apply(lambda x: x.isoweekday())
data = data.drop(['dteday'],axis=1)
data = data.drop(['temp'],axis=1)

#Codificacion variables categoricas
dummy = pd.get_dummies(data['season'], prefix = 'season')
data = pd.concat([data,dummy],axis=1).drop(['season'],axis=1)

#Preparamos los datos para aplicar OnehotEncoder (con la nueva version relativa a la version 0.22 de sklearn es mejor aplicarlo solo sobre las columnas categoricas
# y despues hacer una concatenacion con el resto de datos)
categ_features = ['weathersit','month','weekday']

data_cat = data[categ_features]
data_cat
data_no_cat = data.drop(categ_features,axis=1).drop(['cnt'],axis=1)
data_no_cat

#Preparamos los valores para poder trabajar con Onehot
# recordemos que onehot es una utilidad de sklearn y debe trabajar con arrays de numpy
#features = data.columns.drop(['cnt'])
X_cat = data_cat.values
y = data['cnt'].values

from sklearn.preprocessing import OneHotEncoder

#Esto no funciona con la nueva version de sklearn (0.20)
#enc = OneHotEncoder(categorical_features = [2,8,9], sparse=False, n_values=[3,12,7]) #weathersit, month, weekday

#Para aplicar la nueva version lo mejor es separar un dataframe 
enc = OneHotEncoder(sparse=False) #weathersit, month, weekday
X_cat = enc.fit_transform(X_cat)
print(len(X_cat[0]))
#print('Filas, columnas', X.shape)
X_cat[0]

enc.categories_
#Por ultimo concatenamos la parte categorica con la no categorica
X_no_cat = data_no_cat.values

X = np.concatenate((X_cat,X_no_cat), axis= 1)
len(X[0])
X[0]