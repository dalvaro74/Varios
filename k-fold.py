from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
# Para evitar el wanrning de convergencia
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning) 

iris = datasets.load_iris()
iris['data'].shape


X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=15)

kf = KFold(n_splits=5)

clf = LogisticRegression(penalty='none')
clf.get_params()
#Asi tampoco sale el warning de convergencia (sin necesidad de los imports de arriba)
#clf = LogisticRegression(max_iter=1000)

clf.fit(X_train, y_train)

print(f"El valor de los coeficientes es: {clf.coef_}")
print(f"El valor del intercept es: {clf.intercept_}")

score = clf.score(X_train,y_train)

print("Metrica del modelo", score)

##
preds = clf.predict(X_test)
score_pred = metrics.accuracy_score(y_test, preds)

print("Metrica en Test Ini", score_pred)
##

scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")

print("Metricas cross_validation", scores)

print("Media de cross_validation", scores.mean())

preds = clf.predict(X_test)

score_pred = metrics.accuracy_score(y_test, preds)

print("Metrica en Test", score_pred)