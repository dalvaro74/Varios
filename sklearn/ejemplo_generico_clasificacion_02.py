from sklearn import datasets
from sklearn.model_selection import (cross_val_score, GridSearchCV)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn import tree

diabetes = datasets.load_diabetes()

X = diabetes.data[:150]
y = diabetes.target[:150]
X = pd.DataFrame(X)