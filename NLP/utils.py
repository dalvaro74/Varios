import numpy as np
import pandas as pd
import spacy
import io

from sklearn.model_selection import StratifiedShuffleSplit

nlp = spacy.load('en_core_web_sm', disable= ['ner', 'parser'])

def split_train_test(dataset, split=0.2):

    x, y = zip(*dataset)
    x = np.array(list(x))
    y = np.array(list(y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=1337) #l33t seed
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    splits = {'train':(x_train, y_train), 'test':(x_test, y_test)}
    return splits

# En la funcion preprocess pondremos diferentes opciones de preprocesado del texto
# e iremos probando con distintas combinaciones (comentando y descomentando) para ver
# cual nos genera un mejor modelo.
def preprocess(text, lema = True):    
    
    # Eliminamos los espacios en blanco por delante y detras
    text = text.strip()

    #Tokenizamos
    doc = nlp(text)
    words = [t for t in doc]

    # Quitamos los signos de puntuacion
    #words = [t for t in doc if not t.is_punct]
    
    # Quitamos las stop_words
    #words = [t for t in words if not t.is_stop]

    # Eliminamos palabras de tamaño menor de tres
    #words = [t for t in words if len(t) > 2 and  t.isalpha()]
    
    # Lematizamos
    if(lema):
        words = [t.lemma_ for t in words]
    else:
        words = [t.text for t in words]

    # Ponemos las palabras en minusculas
    words = [t.lower() for t in words]

    # Quitamos las direcciones de correo
    words = list(filter(lambda x:x[0]!='@', words))    

    # Por ultimo eliminamos los caracteres que no queremos que aparezcan
    #all_printables_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    printables_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"\' ?-'
    words = [''.join([c if c in printables_chars else '' for c in word]) for word in words]
    # Generamos el string de salida
    result = ' '.join(words)
    return result.strip()

def get_dataset(df, prep=True, lema=True):

    sentiment_dataset = []
    for index, row in df.iterrows():
        if(index%500==0):
            print(f' {index} elements preprocessed')        
        label = row[1] 
        sentence = row[2]
        # Aqui ira el preprocesado de texto
        if(prep):
            sentence =  preprocess(sentence,lema)
        sentiment_dataset.append((sentence, label))
    return sentiment_dataset