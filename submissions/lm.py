import pandas as pd
from sklearn import linear_model
import pylab as pl
import numpy as np
data = pd.read_csv('predictions.csv')
data = np.exp(data)
Y = data['actual']
del data['actual']

lm = linear_model.LinearRegression(fit_intercept = True)
lm.fit(data, Y)
print('Coefficients: \n', lm.coef_)
print data.shape
print Y.shape
print lm.score(data, Y)


pl.scatter(data['predictions'],Y)
pl.draw()
