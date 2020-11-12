import numpy as np
from sklearn import linear_model

x = np.random.uniform(-2,2,101)
y = 2*x+1 + np.random.normal(0,1, len(x))

#Note that x and y must be in specific shape.

x2 = x.reshape(-1,1)
y2 = y.reshape(-1,1)

x.shape
x2.shape
y2.shape

LM  = linear_model.LinearRegression().fit(x2,y2) #Note I am passing in x and y in column shape

predict_me = np.array([ 9,10,11,12,13,14,15])

predict_me2 = predict_me.reshape(-1,1)

score = LM.score(x2,y2)

predicted_values = LM.predict(predict_me2)

predictions = {'intercept': LM.intercept_, 'coefficient': LM.coef_,   'predicted_value': predicted_values, 'accuracy' : score}
predictions