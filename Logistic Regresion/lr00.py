'''
El dataset Iris contiene valores de 4 caracteristicas:
'sepal length (cm)',
'sepal width (cm)',
'petal length (cm)',
'petal width (cm)'],

En base a esos valores el target tres posibles valores (clases) que son:
'setosa', 'versicolor', 'virginica' y que se corresponden con los valores 0, 1 y 2

Hasta donde mi conocimiento llega la regresion logistica(logit) solo trata con problemas 
binomiales, es decir o es una cosa o no lo es, ya que en el fondo trata con probabilidades. 
Podemos usar la logit para calcular la probabilidad de que para unos datos dados la planta 
sea setosa o no, pero no para que nos de la probabilidad entre tres opciones.
Por tanto, deberia ser labor nuestra separar el problema y los datos en tres problemas 
de regresion distintos.
Afortunadamente y aunque lo haremos en este ejercicio la LogisticRegression sklearn es capaz
de "tragar" con el dataset completo y generar internamente esos tres modelos, por ello al 
sacar los coeficientes y los slpes del modelo obtenemos un vector de tres elementos que se 
corresponde con cada uno de los modelos generados para las tres clases (setosa, versicolor y 
virginica)

En este ejemplo no haremos separcion en train test, porque no nos interesa conocer la bondad 
del modelo, solo su funcionamiento.
'''

from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
# Para evitar el wanrning de convergencia
import warnings
from sklearn.exceptions import ConvergenceWarning
#warnings.simplefilter("ignore", ConvergenceWarning) 

iris = datasets.load_iris()
# Hagamos primero el ejemplo de que le pasamos todos los datos en conjunto
# (sin separar)

X = iris['data']
y = iris['target']

'''
multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
If the option chosen is ‘ovr’, then a binary problem is fit for each label. 
For ‘multinomial’ the loss minimised is the multinomial loss fit across 
the entire probability distribution, even when the data is binary. 
‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ 
selects ‘ovr’ if the data is binary, or if solver=’liblinear’, 
and otherwise selects ‘multinomial’.
'''
# Seleccionamos ovr para que coincida con los resutados de los modelos 
# hechos por separado
# Lo de max_iter es para evitar los problemas de convergencia
# No es necesario en los modelos por separado.
clf = LogisticRegression(max_iter=10000,multi_class='ovr')
clf.fit(X, y)

print(f"El valor de los coeficientes es: {clf.coef_}")
print(f"El valor del intercept es: {clf.intercept_}")
print("\n")

# Ahora generaremos los distintos modelos para cada clase

#diccionario de flores
dic_flowers = {}
for i,name in zip(range(0,3), iris['target_names']):
    dic_flowers[i]=name

for key,value in dic_flowers.items():
    print(f"**********{value}*********")
    y=(iris['target']==key).astype(int)
    clf2 = LogisticRegression(max_iter=10000)
    clf2.fit(X, y)
    print(f"Coeficientes: {clf2.coef_}")
    print(f"Intercept: {clf2.intercept_}")
    print("*****************************")
    print("\n")
