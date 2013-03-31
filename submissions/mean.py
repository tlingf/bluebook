import pandas as pd
import numpy as np
import sys
import csv
from sklearn.metrics import mean_squared_error

fn = sys.argv[1]

i = 0

train_data = pd.read_csv('Train.csv')
train_actual = train_data["SalePrice"]
print "mean of Train", np.mean(train_actual)
#train_actual = np.log(train_actual)

predictions = pd.read_csv(fn)
predictions = predictions["SalePrice"]
#csv_file_object = csv.reader(open(fn,'rb'))
#header = csv_file_object.next()
#col = header.index("prediction")
#print "prediction col", col
#predictions = []

# 401126 rows incl header
#for row in csv_file_object:
#  if i < 401125:
#    predictions.append(float(row[col]))
#    if i < 10: print row[col]
#  i += 1

# combine train data with predictions, get year


# This matches
#mse = mean_squared_error(train_actual, predictions)
#rmse = np.sqrt(mse)
#print "rmse", rmse
print "mean of predictions:", np.mean(predictions)
#print "mean of predictions:", np.mean([np.exp(x) for x in predictions if x > 0])
#print "mean of actual", np.mean(data["SalePrice"])

print "diff of predictions and actual", np.mean(predictions) - np.mean(train_actual)

data = pd.read_csv('../kaggle/bluebook-analysis/Submissions/submit_rf_4.csv')
print "mean of best submission", np.mean(data["SalePrice"])
print np.mean(data["SalePrice"]) - np.mean(train_actual)
