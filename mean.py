import pandas as pd
import numpy as np
import sys

fn = sys.argv[1]
data = pd.read_csv(fn)

print data
print "mean", np.mean(data["SalePrice"])

data = pd.read_csv('Train.csv')
print "mean of Train", np.mean(data["SalePrice"])
