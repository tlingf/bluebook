""" Bluebook Kaggle
Author: Ling
Date Started: 2/14/13"""

import numpy as np
import csv as csv
#from sklearn import svm
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.cross_validation import KFold
#from sklearn import linear_model
import datetime
import string
from sklearn import preprocessing
#from scipy.spatial import distance

print datetime.datetime.today()
f = open('Train.csv', 'rb')
csv_file_object = csv.reader(f) #Load in the training csv file
header = csv_file_object.next() #Skip the fist line as it is a header

print "Opened csv"
#train = csv.DictReader(open('Train.csv', 'rb')) #Load in the training csv file
#header = csv_file_object.next() #Skip the fist line as it is a header
#test = csv.DictReader(open('Valid.csv', 'rb'))

print "Creating data from csv"

def write_csv_to_array(csv_obj):
    i = 0
    data = []
    for row in csv_obj: #Skip through each row in the csv file
        if i < 500000:
            data.append(row) #adding each row to the data variable
        i += 1
    data = np.array(data) #Then convert from a list to an array
    return data

def get_date(date_column, num_rows):
    date_data = []
    for i in xrange(num_rows):
        dt_str = date_column[i]
        d = datetime.datetime.strptime(dt_str, '%m/%d/%Y %H:%M')
        #if i< 10:
            #print d
        date_data.append([d.year, d.month, d.day])
    #print date_data
    return date_data


train = write_csv_to_array(csv_file_object)
del csv_file_object
f.close()



Y = train[::,1]
train = np.delete(train, 1,1) # Delete the price column
gc.collect()
header.remove("SalePrice") # This is Y

csv_test = csv.reader(open('Valid.csv','rb'))
header_test = csv_test.next()
test = write_csv_to_array(csv_test  )
del csv_test

productSize = header.index("ProductSize")
saledate = header.index("saledate")
salesId = header.index("SalesID")

def clean_data(data):
    data_out = {}
    num_rows = len(data)
    
    d = {"Mini":0,"Compact":1,"":2,"Small":3,"Large":4,"Medium":5,"Large / Medium":6}
    for i in xrange(num_rows):
        data[i,productSize] = d[data[i,productSize]]
    #data[productSize] = [[d[x] for x in data[::,productSize]]]
    
    date_data = get_date(data[0::,saledate], num_rows)
    data = np.delete(data,saledate,1)
    data = np.column_stack([data,date_data])
    
    return data


print "Cleaning Train Data"
train = clean_data(train)
print "Cleaning Test Data"
test = clean_data(test)

# Update header to date shape changes
header.remove("saledate")
header.extend(["Sale Year", "Sale Month", "Sale Day"])

productSize = header.index("ProductSize")

print "shape of train data", train.shape
print "shape of test data", test.shape
num_of_columns = train.shape[1]

train_rows = len(train)
test_rows = len(test)
for col_num in xrange(num_of_columns):
    
    if col_num not in [productSize]:
        print col_num, header[col_num]
        try:
            #print test[0,2]
            #print train[0,2]
            train[0::,col_num] = train[0::,col_num].astype(np.float)
            #print test[0,col_num]
            test[0::,col_num] = test[0::,col_num].astype(np.float)
        
        # If non number values
        except ValueError:
            mapping = enumerate(np.unique(np.append(train[::,col_num], test[::,col_num]))) # Unique values
            d = {}
            for x in mapping:
                d[x[1]] = x[0]
            #d = {x[1]:x[0] for x in mapping}
            
            for i in xrange(train_rows):
                #if i < 10: print d[train[i, col_num]]
                train[i, col_num] = d[train[i, col_num]]
                
            for i in xrange(test_rows):
                test[i, col_num] = d[test[i, col_num]]

X = train.astype(np.float)
Y = Y.astype(np.float)
test = test.astype(np.float)

forest = RandomForestRegressor(n_estimators=10,n_jobs=1, oob_score = True) # compute_importances=True, 
forest = forest.fit(X,Y)
print "Forest Score", forest.score(X,Y)
output_forest = forest.predict(test)
print "oob score", forest.oob_score_