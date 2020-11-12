#%matplotlib inline

import numpy as np
import pandas as pd
import spacy
import io

from time import time

import pickle
import json
import os
import csv

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from random import sample

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils import *


#Cargamos el Dataset
path1 = 'NLP/data/train_sentiment.csv'
path2 = './data/train_sentiment.csv'
df = pd.read_csv(path1,sep=',', decimal='.',encoding='ISO-8859-1')
df

#Establecemos un dataset mas pequeño para hacer pruebas
df_smaple = df[0:10000]

# Obtenemos el array con los textos del dataset y llevamos a cabo el preproceso de los textos
# Ponemos un timer para ver cuanto el proceso (casi 10 minutos)
start_time = time()
sentiment_dataset = get_dataset(df_smaple, True, True)
elapsed_time = time() - start_time
print("Preprocess Elapsed time: %0.10f seconds." % elapsed_time)
print(sentiment_dataset[0])
len(sentiment_dataset)

#Guardamos el dataset en disco
#full_airbnb_images.to_csv(join(PROJECT_PATH,'data/airbnb-images.csv'), sep=';', decimal='.', index=False)

split = split_train_test(sentiment_dataset, 0.20)


#Tunning de Hyperparametros y entrenamiento de modelos

#*********************
#Bayes
#*********************

pipeline_bayes = Pipeline([
    ('vect', CountVectorizer(max_df=0.5)),
    #('tfidf', TfidfTransformer()), #No vale la pena usar esta normalizacion porque no se consiguen mejores resultados
    ('clf', BernoulliNB()),])

pipeline_bayes.get_params().keys() # que parametros podemos tocar en el gridsearch!

param_grid_bayes = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 2), (1, 3), (2, 3)),  #ngrams to test
    'vect__analyzer': ('word', 'char', 'char_wb'),
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.1, 1.0)    
}

grid_search_bayes = GridSearchCV(pipeline_bayes, param_grid_bayes,cv=5, n_jobs=-1, verbose=1)

grid_search_bayes.fit(split['train'][0], split['train'][1])
best_parameters = grid_search_bayes.best_estimator_.get_params()
for param_name in sorted(param_grid_bayes.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
print("Best score: %0.3f" % grid_search_bayes.best_score_)

pipeline_bayes.set_params(**best_parameters)
predictions = pipeline_bayes.score(split['test'][0], split['test'][1])
print('TEST SCORE BAYES: {}'.format(predictions))

#*********************
#SVC
#*********************

pipeline_svc = Pipeline([
    ('vect', CountVectorizer(max_df=0.75,ngram_range=(2,3),analyzer='char_wb')),
    ('tfidf', TfidfTransformer(norm='l1',use_idf=True)), #Sin esta normalizacion obtenemos mejores resultados
    ('clf', SVC()),])
pipeline_svc.get_params().keys() # que parametros podemos tocar en el gridsearch!

# Para el caso de SVC
# Hemos comprobado que los tiempos de entrenamiento para este algoritmo son muy elevados
# Por ello usaremos los parametros ya calculados para Benoulli y solo haremos tunning con los
# propios de SVC 
param_grid_svc = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__ngram_range': ((1, 2), (1, 3), (2, 3)),  #ngrams to test
    #'vect__analyzer': ('word', 'char', 'char_wb'),
    #'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__C': (0.1, 1.0),
    'clf__gamma':(0.001, 0.0001)
}
grid_search_svc = GridSearchCV(pipeline_svc, param_grid_svc,cv=5, n_jobs=-1, verbose=1)

grid_search_svc.fit(split['train'][0], split['train'][1])
best_parameters = grid_search_svc.best_estimator_.get_params()
for param_name in sorted(param_grid_svc.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
print("Best score: %0.3f" % grid_search_svc.best_score_)

pipeline_svc.set_params(**best_parameters)
predictions = pipeline_svc.score(split['test'][0], split['test'][1])
print('TEST SCORE SVC: {}'.format(predictions))