import pandas as pd
import numpy as np
import sys
import csv
from sklearn.metrics import mean_squared_error

fn = sys.argv[1]

predictions = pd.read_csv(fn)


#submission['SalesID'] = predictions['SalesID']
#predictions = predictions["SalePrice"]



rf = pd.read_csv('submit_rf400K.csv')
rf = rf['SalePrice']

print len(rf)
print len(predictions)

n = []
for i in xrange(len(predictions)):
    n.append(.5*predictions["SalePrice"][i] + .5*rf[i])
print len(n)
submission = pd.DataFrame({'SalesID': predictions['SalesID'], 'SalePrice': pd.Series(n)})
print len(submission)

submission[['SalesID','SalePrice']].to_csv('submit_rf_gbm_run13.csv')
