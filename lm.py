import pandas as pd
from sklearn import linear_model
import pylab as pl

data = pd.read_csv('predictions.csv')
Y = data['actual']
del data['actual']

lm = linear_model.LinearRegression(fit_intercept = True)
lm.fit(data, Y)
print('Coefficients: \n', lm.coef_)
print lm.score(data, Y)

pl.scatter(data,Y)
