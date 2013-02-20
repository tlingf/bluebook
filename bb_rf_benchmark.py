from collections import defaultdict
import numpy as np
import pandas as pd
pd.set_option('use_inf_as_null', True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
#from PyWiseRF import WiseRF
from sklearn import preprocessing
import util
#import gc
import datetime
import re
import string
from time import time
from dateutil.relativedelta import relativedelta
import numpy.ma as ma
print datetime.datetime.today()
import csv
import sys

comment = sys.argv[1]
# Latest
# fix tire size, when run on all data had issues
# done - changed variables in cleaning_data all to new_arr to avoid storing extra var
# change na stick_length to mean, shouldn't be big diff
# tried and no - only take certain mfg Id's? 26 (caterpillar) and 74, and the top count ones

# to do
# figure out avg values of na values for all col's
# why is stick length mean so high

testing = 0
run_ml = 1
use_wiserf =  0 # Doesn't work
#run_ml = 1
t1 = time()

#import csv
#csv_out = 

def get_date_dataframe(date_column):
    return pd.DataFrame({
       # "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column],
        "SaleWkDay": [d.isocalendar()[2] for d in date_column],
        # "SaleYrWk": [d.isocalendar()[1] for d in date_column],
        "SaleYearMo3":[str((d + relativedelta(months = -3)).year) +
                       "-" + str.zfill(str((d + relativedelta(months = -3)).month),2) + "-01" for d in date_column],
        "SaleYearMo2":[str((d + relativedelta(months = -2)).year) +
                       "-" + str.zfill(str((d + relativedelta(months = -2)).month),2) + "-01" for d in date_column]
        #}, index=date_column.index)
        })

def map_external_data(data):
    """ Map year of sale to economic measures that year, from external data"""
    ext_data = util.get_external_data()
    #tcons_mean = np.mean(ext_data.values())
    tcons_min = min(ext_data["TTLCONS"])
    tsales_median = np.median(ext_data["HTRUCKSSA"]) # try mean
    cpi_min = min(ext_data["TCPI"])
    #print ext_data["CPI"]
    #print "sample cpi", ext_data["CPI"]["2006-08-01"]
    
    ext_data["TTLCONS"] = ext_data["TTLCONS"].fillna(tcons_min)
    ext_data["HTRUCKSSA"] = ext_data["HTRUCKSSA"].fillna(tsales_median) # is not much smaller in earlier years
    ext_data["TCPI"] = ext_data["TCPI"].fillna(0) # cpi_min
    
    # Get the data value (x) corresponding to column for each date in data
    data["tcons"] = [ext_data["TTLCONS"][x] for x in data["SaleYearMo3"]]
    data["tsales"] = [ext_data["HTRUCKSSA"][x] for x in data["SaleYearMo3"]]
    data["tcpi"] = [ext_data["TCPI"][x] for x in data["SaleYearMo2"]]
    data["cpi"] = [ext_data["CPI"][x] for x in data["SaleYearMo2"]]
    del data["SaleYearMo3"]
    del data["SaleYearMo2"]
    return data
    #d = datetime.datetime.strptime(dt_str, '%m/%d/%Y %H:%M')

def get_auction_avg():
    """ Get the average price for each auction"""
    # all the dates
    auction_means = {}
    uniq_dates = np.unique(x for x in np.append(train["salesdate"].values,test[col].values))
    uniq_states =np.unique(x for x in np.append(train["state"].values,test[col].values))
    uniq_auctioneerid =np.unique(x for x in np.append(train["auctioneerID"].values,test[col].values))
    for date in uniq_dates:
        for state in uniq_states:
            for auc in uniq_auctioneerid:
                index = date + "|" + state + "|" + auc
                auction_means[index] = np.mean(data[data['salesdate']==date & \
                                                    data["state"] == state & data["auctioneerID"] == auc])
                
def map_columns(col, d, data, data_out):
    """ Change categorical to number.. not binarized"""
    data[col]=data[col].fillna(value ="")
    mapping = pd.Series([d[x] for x in d.keys()], index = d.keys())
    data_out[col] = data[col].map(mapping)
    #data_out = data_out.join(data[col].map(mapping))
    #data.drop([col], axis =1)
    return data_out

col_list = ["ProductSize","UsageBand",'fiProductClassDesc'] # "ProductGroup"
# referenced in clean_columns and final col parsing

def clean_columns(data, data_out):
    appendix = util.get_appendix()
    
    # Reduces storage?
    data = data.replace("None or Unspecified","None")
    
    drop_list = ["ProductGroupDesc"] #"fiModelDesc"]
    #ProductGroupDesc is just long desc for ProductGroup, repeat
    # fiModelDesc is repeat, and described in fiBaseModel + fiSecondaryModel + fiModelSeries
    # fiProductClassDesc is too long, e.g. "Wheel Loader - 110.0 to 120.0 Horsepower"
    data = data.drop(drop_list,axis =1)
    num_rows = data.shape[0]
    columns = set(data.columns)
    for col in columns:
        #print "Cleaning", col
        #if col == "ProductGroup":
            #d = {"SSL":0, "BL":1, "TEX":2,"TTT":3,"WL":4,"WG":5,"":0}
        
        #Age of machine
        if col == "YearMade":
            #print "calculating years age"
            new_arr =[]
            year_new_arr = []
            for i in xrange(num_rows):
		# fix 2014 yearmade
    	        yearmade = min(data["saledate"][i].year, data[col][i])
                age = (data["saledate"][i].year - yearmade)
                new_arr.append(age)
    		year_new_arr.append(yearmade)
            data['YearsAge'] = new_arr
	    data[col] = year_new_arr
            #del data["YearMade"]
	    # might need to use the new mfg year later
        
        elif col == "ProductSize":
            # ProductSize corresponds to type, so this is ok
            #d = {"Mini":0,"Compact":1,"":2,"Small":3,"Large":4,"Medium":5,"Large / Medium":6}
            d = {"Mini":0,"Compact":1,"Small":2,"":3,"Medium":4,"Large / Medium":5,"Large":6}
            data_out = map_columns(col, d, data, data_out)
        
        elif col == "Tire_Size":
            data[col]=data[col].fillna(value =0)
            new_arr = []
            for x in data[col]:
                if x != 0 and x == "None":
                    repl = string.replace(string.replace(str(x), "\"",""), "'","")
                    try: new_arr.append(float(repl))
                    except: new_arr.append(0)
                else: new_arr.append(0)
            data[col] = new_arr
            # is float64 really better?
            #data[col] = pd.Series([string.replace(string.replace(str(x), "\"",""), "'","") for x in data[col]])
        
        elif col == "Blade_Width":
            data[col]=data[col].fillna(value =0)
            #data[col]=data[col].fillna(value =0)
            new_arr = []
            for x in data[col]:
                if x == "<12'": repl = 0
                elif x == "None": repl = 13.5 # calculated average for MG ProductGroup (only relevant one)
                else: repl = float(string.replace(str(x), "'",""))
                new_arr.append(repl)
            data[col] = new_arr
        
        elif col == "Stick_Length":
            # Avg is close to None and na
            new_arr=[]
            for x in data[col]:
                i =0
                if pd.isnull(x) or x == "None": l = 0
                else:
                    repl = string.replace(string.replace(str(x), "\"",""), "'","")
                    repl = string.split(str(repl)," ")
                    ft , inches = repl[0], repl[1]
                    #if i < 10: print repl
                    l = 12*float(ft) + float(inches)
                new_arr.append((int(round(l,1))))
            data[col] = new_arr # pd.Series(new_arr, dtype = 'float16')
        
        elif col == "Undercarriage_Pad_Width":
            data[col]=data[col].fillna(value =0)
            #print data[col][0:10]
            #print data[col]
            s = np.unique(x for x in data[col])
            #print "unique values for undercarriage pad width", s
            new_arr = []
            for x in data[col]:
                repl = string.replace(str(x), " inch","")
                try:new_arr.append(int(round(float(repl),0)))
                except:
                    new_arr.append(0)
            #data[col] = pd.Series([string.replace(str(x), " inch","") for x in data[col] ])
            data[col] = new_arr
        # to do
        # Stick_Length : 22 : ['nan' '10\' 10"' '10\' 2"' '10\' 6"' '11\' 0"' '11\' 10"' '12\' 10"'

        elif col == "Backhoe_Mounting":
            # fillna doesn't really work here?
            new_arr = []
            for x in data[col]:
                if x == "None": new_arr.append(x)
                elif x != "Yes": new_arr.append("")
                else:new_arr.append("Yes")
                #arr.append("") if x != "Yes" else "Yes"
            data[col] =new_arr
            #data[col] = ["" for x in data[col] if x != "Yes"]
        
        # Create letters only model
        if col == 'fiBaseModel':
            data[col]=data[col].fillna(value ="")
            model_letters = []
            for x in data[col]:
                # Find letters only at beginning of name
                try:
                    m = re.search('[a-zA-Z]',x)
                    if m is not None:
                        model_letters.append(m.group(0))
                    else: model_letters.append("")
                except TypeError: # Not sure what this is
                    model_letters.append("")
            data['fiBaseModelL'] = model_letters
        
        # Get manufacturing ID based on MachineID
        if col == "MachineID":
            error_count =0
            
            mfgids = []
            mfgids2 = []
            power_max_l = []
            power_min_l = []
            power_unit_l = []
            for x in data[col]:
                try:
                    appendix_match = appendix[x]
                    mfgid_temp, power_u, power_min, power_max = tuple(appendix_match)
                    #if mfgid_temp  in [0,26, 43, 25, 103, 121, 74, 55, 92, 99, 176, 750, 135, 54, 166, 95, 46, 158, 86, 405]: # most freq, > 1K
                        #mfgid = mfgid_temp
                        #mfgid2 = 0
                    #else:
                        #mfgid = 0
                        #mfgid2 = mfgid
                    if power_max == "" and power_min == "":power_max, power_min = 0,0
                    elif  power_max == "" or pd.isnull(power_max): power_max = 0
                    elif power_max == 1000000: # when says e.g. "100+"
                        power_max = power_min
                    if power_min == 0: power_min = power_max
                    elif power_min == "":power_min = -1
                except KeyError: mfgid, power_u, power_min, power_max = 0, "", 0, 0
                #mfgids.append(mfgid) # default to 0
                #mfgids2.append(mfgid2)
                mfgids.append(mfgid_temp)
                try:
                    power_max_l.append(float(power_max))
                    power_min_l.append(float(power_min))
                except:
                    print "error", power_max
                    raise KeyboardInterrupt
                power_unit_l.append(power_u)
            #mfgid = [appendix[x] for x in data[col]] 
            data['MfgID'] = mfgids
            #data['MfgID2'] = mfgids2
            data['power_max'] = power_max_l
            data['power_min'] =  power_min_l
            data['power_u'] =  power_unit_l
            #data[["power_min", "power_max"]].to_csv('power_min_out', index=True)
            # too slow
            # Don't know a better way to do this
            #data['power_max'] = power_max_l
            #data['power_min'] =  power_min_l
            # Guess missing values based on 
            #for x in xrange(num_rows):
            #    if power_max_l[x] == "":                    
            #        # Find all vaues where Year Made, ProductGroup, manufacturer are same
            #        similar_max = data[ (data['ProductGroup'] == data.xs(x)["ProductGroup"]) &
            #                           (data['YearMade'] == data.xs(x)["YearMade"] ) & (data['power_max'] != "")
            #                          ]["power_max"].astype(np.float)
            #        power_max_l[x] = np.median(similar_max)
            #    if power_min_l[x] == "":
            #        similar_min = data[ (data['ProductGroup'] == data.xs(x)["ProductGroup"]) &
            #                           (data['YearMade'] == data.xs(x)["YearMade"] )& (data['power_min'] != "")
            #                          ]["power_min"].astype(np.float)
            #        power_min_l[x] = np.median(similar_min)
            #max_avg = np.median(ma.masked_values(power_max_l,""))
            #min_avg = np.median(ma.masked_values(power_min_l, ""))
        
        # Less Important
        if col == "UsageBand":
            d = {"Low":0, "":1,"Medium":2,"High":3}
            data_out = map_columns(col, d, data, data_out)
            
            # Old Method
            #m = [(v, k) for k, v in d.iteritems()]
            #mapping = pd.Series([x[0] for x in m], index = ["Low", "", "Medium", "High"])
    
    data = data.drop(col_list,axis =1)
            
    return data, data_out
            
    #if data["UsageBand"]

def binarize_cols(col, train, test, train_fea, test_fea):
    """ Change categorical variables to binary columns"""
    lb_bm = preprocessing.LabelBinarizer()
    lb_bm.fit(np.append(train[col].values, test[col].values))
    train_bm = lb_bm.transform(train[col])
    test_bm = lb_bm.transform(test[col])
    
    # Join them together
    classes = lb_bm.classes_
    for i in xrange(len(classes)):
        category = classes[i]
        train_fea[col + "-" + str(category)] = train_bm[0::,i]
        test_fea[col + "-" + str(category)] = test_bm[0::,i]
    return train_fea, test_fea

def data_to_fea():
    """Main preprocessing"""
    train, test = util.get_train_test_df()
    
    #train = train[train["saledate"] >= datetime.datetime(1993,1,1)]
    #train = train.reindex(range(len(train)))
    
    train_fea = get_date_dataframe(train["saledate"])
    test_fea = get_date_dataframe(test["saledate"])

    train_fea = map_external_data( train_fea)
    test_fea = map_external_data(test_fea)

    print "Cleaning Columns"
    train, train_fea = clean_columns(train, train_fea)
    test, test_fea = clean_columns(test, test_fea)
    # train[["power_min", "power_max", "SalePrice", "ProductGroup", "YearMade"]].to_csv('power_min_out.csv', index=True)
    
    columns = set(train.columns)
    print columns
    columns.remove("SalesID")
    columns.remove("SalePrice")
    columns.remove("saledate")
    for col in columns:
      #try:
        # ignore these ones  ["ProductGroup","ProductSize","UsageBand"]
        # these deleted already["ProductGroupDesc", "fiProductClassDesc"]
        #if 0:
        #if col == "Coupler_System":
        #if col in ['fiBaseModel', 'fiModelDesc']: # Testing
        if col not in col_list : # REAL ONE
            
        # col_list = ["ProductSize","UsageBand",'fiProductClassDesc'] # "ProductGroup"
        #if col == "Backhoe_Mounting": # Testing - error in the fillna "convert string to float" but why?
            # error with BladeExtension
            #print test[col][0:25]
            print "starting", col
            if col == 'fiBaseModel': # Special case
                train[col] = [str(x).strip() for x in train[col].values]
                test[col] = [str(x).strip() for x in test[col].values]
            
            if col  in [ 'fiBaseModelL', 'ProductGroup', 'fiSecondaryDesc', 'fiSecondaryDesc', 'state',
                        'auctioneerID', 'power_u', 'MfgID', 'Enclosure', 'SaleMonth', "SaleWkDay"] :
            #if col in ['fiBaseModelL', 'ProductGroup', 'fiSecondaryDesc', 'fiSecondaryDesc', 'Enclosure', 'MfgID']:
                print "binarize", col
                if col in ['auctioneerID', 'MfgID']:
                    train[col]=train[col].fillna(value =0)
                    test[col] = test[col].fillna(value = 0)
                else:
                    train[col]=train[col].fillna(value ="")
                    test[col] = test[col].fillna(value = "")

                # would binarize BaseModel but too much memory  
                train_fea, test_fea = binarize_cols(col, train, test, train_fea, test_fea)
                
            elif train[col].dtype == np.dtype('object') and col not in ['power_min', 'power_max']:
                #print "filling na"
                train[col]=train[col].fillna(value ="")
                test[col] = test[col].fillna(value = "")
                #
                #print "counting unique"
                s = np.unique(x for x in np.append(train[col].values,test[col].values))
                print  col, ":", len(s), ":", s[:10]
                
                # Binarize these ones:
                if len(s) >2 and len(s) < 100: # in [ 'fiBaseModel']
                # try on len(s) > 3
                # don't binarize datasource (6 values), lower performance
                    print "binarize", col
                    # would binarize BaseModel but too much memory  
                    train_fea, test_fea = binarize_cols(col, train, test, train_fea, test_fea)
                
                # Just enumerate these
                else:
                #if 1:
                    print "enumerate",col
                    #Don't need below line for full data set usually
                    if test[col].dtype != np.dtype(object):
                        print "changing test obj type"
                        test[col] = test[col].astype(object) # in case test had diff type
                    
                    if len(s) == 2:
                        # assume one is ""
                        # one replace function would have been able to do this
                        new_arr_train = []
                        new_arr_test = []
                        for x in train[col]:
                            repl = 0 if x == "" else 1
                            new_arr_train.append(repl)
                        for x in test[col]:
                            repl = 0 if x == "" else 1
                            new_arr_test.append(repl)
                        train_fea[col] = new_arr_train
                        test_fea[col] = new_arr_test
                    else:
                        # Regular dumb indexing
                        mapping = pd.Series([x[0] for x in enumerate(s)], index = s) # Original code
                        
                        # Faster method to add col
                        print "mapping col"
                        
                        
                        train_fea[col] = train[col].map(mapping)
                        test_fea[col] = test[col].map(mapping)
                        #train_fea[col] = np.log(train_fea[col]+1)
                        #test_fea[col] = np.log(test_fea[col]+1)
                    if pd.isnull(train_fea[col]).any() or pd.isnull(test_fea[col]).any():
                        print "HAS NAN", col
                    
            else:
            # Numeric field
            
                print col, " as number"
            #if train[col].dtype != np.dtype('object'):
                #m = np.mean([x for x in train[col] if not np.isnan(x)])
                
                # can use pd.isnull(frame).any() TRY 
                if col =="MachineID" or col == "ModelID" or col == "datasource":
                    m = 0 # value to fill
                elif col in [ "MachineHoursCurrentMeter", 'Stick_Length']:
                    # Diff w/ 0 is not large
                    train_m = round(np.mean([x for x in train[col] if x > 0]),1)
                    test_m = round(np.mean([x for x in test[col] if x > 0]),1)
                #elif col == "Stick_Length":
                    #train_m = round(np.median([x for x in train[col] if x > 0]),1)
                    #test_m = round(np.median([x for x in test[col] if x > 0]),1)
                else:
                    # Calculate median, better oob performance than mean
                    train_m= np.median(train[col]) # if x > 0?
                    test_m= np.median(test[col])
                #print m
                #if col == 'power_min':
                #    train[col] = train[col].fillna(value =0)
                #    test[col] = test[col].fillna(value=0)
                #    
                #    train_fea[col] = train[col]
                #    test_fea[col] = test[col]
                #    #test_fea[col] = test[col]
                #    print "blank for power min", len(train[train["power_min"] == 0])
                #    train_fea = train_fea[train_fea["power_min"] != 0] # try this out # now at the top
                #    train = train[train["power_min"] != 0] # try this out # now at the top
                #    #test = test[test["power_min"] != 0]
                    #test_fea = test_fea[test_fea["power_min"] != 0]
                    
                    #print "reindexing"
                    #
                    #train_len = train_fea.shape[0]
                    #print "train len", train_len
                    #train = train.reindex(range(train_len))
                    #train_fea = train_fea.reindex(range(train_len))
                    
                if col in ["Stick_Length"]:
                    train[col] = train[col].replace(0,train_m)
                    test[col] = test[col].replace(0,test_m)
                # maybe special case for power_min when 0 (nan) but this isn't common
                else:
                    train[col] = train[col].fillna(value =train_m)
                    test[col] = test[col].fillna(value=test_m)
                print col, train_m
                
                
                # "mean is nan" 
                if np.isnan(train_m): train[col] = train[col].fillna(value =0)
                if np.isnan(test_m):test[col] = test[col].fillna(value=0)
                
                #print "converting to float"
                #train[col] = train[col].astype('float64')
                #test[col] = test[col].astype('float64')
                #train[col] = np.log(train[col]+1)
                #test[col] = np.log(test[col]+1)
                
                train_fea[col]  = train[col]
                test_fea[col] = test[col]
                
                #if col != "power_min":
                #    del train[col]
                #    del test[col]
                #train_fea = train_fea.join(train[col])
                #test_fea = test_fea.join(test[col])
                if pd.isnull(train_fea[col]).any() or pd.isnull(test_fea[col]).any():
                    print "HAS NAN", col
        
      #except:
            #print "Error with col", col

    
    #print "train fea"
    #print train_fea
    
    # This isn't really necessary
    train_fea=train_fea.fillna(method='pad')
    test_fea = test_fea.fillna(method ='pad')
    #train_fea = train_fea[train_fea["SaleYear"] >= 1993] # try this out # now at the top
    #train = train[train["saledate"] >= datetime.datetime(1993,1,1)] # try this out # now at the top
    train = train.drop(["saledate"], axis = 1)
    return train_fea, test_fea, train["SalePrice"]


train_fea, test_fea , train_Y = data_to_fea()
train_Y = np.log(train_Y)

train_len = train_fea.shape[0]
print "new train length", train_len
train_len = len(train_Y)
print "new train length", train_len

#test_fea = test_fea.astype('float64')
if use_wiserf == 1:
    rf = WiseRF(n_estimators=50) # Doesn't work right now
elif run_ml == 1:
    print "SK Learn Running Forest"
    if testing == 1: rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, compute_importances = True, oob_score = True)
    else: rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, compute_importances = True, oob_score = True)
    #print "Learn Gradient Boosting" # Slower
    #rf = GradientBoostingRegressor(subsample = .67) #n_estimators = 100 by default # Cannot parallelize
    
    rf.fit(train_fea, train_Y)
    
    print "Fitting"
    predictions = rf.predict(test_fea)
    predictions = np.exp(predictions)
#print "Mean Squared Error:", np.sqrt(mean_squared_error(train_Y, rf.predict(train_fea)))
util.write_submission("submit_rf" + comment + ".csv", predictions)
imp = sorted(zip(train_fea.columns, rf.feature_importances_), key=lambda tup: tup[1], reverse=True)
logger = open("out/rf_log.txt","a")

csv_w = csv.writer(open('out/rf_features' + comment + '.csv','wb'))
for fea in imp:
    csv_w.writerow([fea[0],fea[1]])

for fea in imp:
    if testing == 0:
        if fea[1] > 0.000001:
            print fea[0], "|", fea[1]
    else:
        if fea[1] > 0.01:
            print fea[0], "|", fea[1]
            
print "oob score:", rf.oob_score_
print "score", rf.score(train_fea, train_Y)
logger.write("\n" + comment+ "\n")
logger.write("oob score:" +str( rf.oob_score_)+ "\n")

print datetime.datetime.today()

def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

# RMSE
print "Calculating RMSE"
#
train_predict = rf.predict(train_fea)
train_Y = np.exp(train_Y)
train_predict = np.exp(train_predict)
#print train_Y

#print "reindexing"
#train_len = len(train_predict)
#train_Y = train_Y.reindex(range(train_len))
##train_fea = train_fe.reindex(train_len)
                    
                    
train_len = len(train_predict)
error_sum = 0
i =0 
for i in xrange(train_len):
    error_unit = (np.log(train_Y[i] ) - np.log(train_predict[i] ))**2
    #if is_numeric(train_Y[i]):
        #print train_Y[i]
    #try:error_unit = (np.log(train_Y[i] ) - np.log(train_predict[i] ))**2
    #except:
        #print train_Y[i], train_predict[i]
    error_sum += error_unit
rmse = np.sqrt(error_sum/float(train_len))
print "RMSE:", rmse

print datetime.datetime.today()
t2 = time()
t_diff = t2-t1
print "Time Taken (seconds):", round(t_diff,0)
logger.write("RMSE:" + str(rmse)+ "\n")
logger.write("Time:" + str(round(t_diff,0))+ "\n")
