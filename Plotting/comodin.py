import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.svm import NuSVC

np.random.seed(2018)

# generate random training features
X = np.random.random((100, 10))

# class labels
y = np.random.randint(2, size=100)

clf = NuSVC(nu=0.4, gamma='auto')

# Compute score for one parameter combination
grid = GridSearchCV(clf,
                    cv=StratifiedKFold(n_splits=10, random_state=2018),
                    param_grid={'nu': [0.4]},
                    scoring=['f1_macro'],
                    refit=False)

grid.fit(X, y)
print(grid.cv_results_['mean_test_f1_macro'][0])

# Recompute score for exact same input
result = cross_validate(clf,
                        X,
                        y,
                        cv=StratifiedKFold(n_splits=10, random_state=2018),
                        scoring=['f1_macro'])

date_info = {'year': "2020", 'month': "01", 'day': "01"}
print(*date_info)
x,y = zip(*date_info.items())

params_fixed = {'silent': 1,'base_score': 0.5,'reg_lambda': 1,
'max_delta_step': 0,'scale_pos_weight':1,'nthread': 4,
'objective': 'binary:logistic'}

def add(a=0, b=0):
    return a + b
d = {'a': 2, 'b': 3}
add(**d)

# Hagamos la prueba con una lista de listas de tres elementos
list_tmp1 = [[1, 'x', 11], [2, 'y', 22], [3, 'z', 33]]

p,q, r = zip(*list_tmp1)
q