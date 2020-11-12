import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import datasets
iris = datasets.load_iris()

type(iris.data)


X_train , X_test, y_train ,y_test =train_test_split(iris.data, iris.target, random_state=0)

