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
        "SaleYear": [d.year for d in date_column],
        #"SaleMonth": [d.month for d in date_column],
	#"SaleWkDay": [d.isocalendar()[2] for d in date_column],
        "SaleDay": [d.day for d in date_column],
        
        # "SaleYrWk": [d.isocalendar()[1] for d in date_column],
        "SaleYearMo3":[str((d + relativedelta(months = -3)).month) +
                       "/1/" + str((d + relativedelta(months = -3)).year) for d in date_column],
        "SaleYearMo2":[str((d + relativedelta(months = -2)).month) +
                       "/1/" + str((d + relativedelta(months = -2)).year) for d in date_column]
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
    #data["tcons"] = [np.log(ext_data["TTLCONS"][x]) for x in data["SaleYearMo3"]]
    #data["tsales"] = [np.log(ext_data["HTRUCKSSA"][x]) for x in data["SaleYearMo3"]]
    data["tcpi"] = [np.log(ext_data["TCPI"][x]) for x in data["SaleYearMo2"]]
    #data["cpi"] = [np.log(ext_data["CPI"][x]) for x in data["SaleYearMo2"]]
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

# These already added to the "out"
col_list = ["ProductSize", "UsageBand"] #,'fiProductClassDesc'] # ,"UsageBand",'fiProductClassDesc'] # "ProductGroup"
# referenced in clean_columns and final col parsing

def clean_columns(data, data_out):
    appendix = util.get_appendix()
    
    # Reduces storage?
    data = data.replace("None or Unspecified","None")
    
    drop_list = ["ProductGroupDesc"]# ,"UsageBand"] #"fiModelDesc"]
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
		# fix datapoints with bad yearmade (2014) or saledate < yearmade
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
	elif col == "saledate":
	    #print col
	    data["SaleMonth"]= [d.month for d in data[col]]
	    data["SaleWkDay"]= [d.isocalendar()[2] for d in data[col]]
	
	# Better as binary value, not converted to value
	#elif col == "Tire_Size":
	#	data[col]=data[col].fillna(value =0)
	#	new_arr = []
	#	print data[col]
	#	for x in data[col]:
	#	    if x != 0 and x == "None" and not pd.isnull(x):
	#		repl = string.replace(string.replace(str(x), "\"",""), "'","")
	#		try: new_arr.append(float(repl))
	#		except: new_arr.append(0)
	#	    else: new_arr.append(0)
	#	mean = np.median([x for x in new_arr if x > 0])
	#	print "tire size mean", mean
        #       data[col] = new_arr
	#	data[col] = data[col].replace(0,mean)
	#	print "Tire_Size", data[col]
	
	if 1:
	    
		# is float64 really better?
		#data[col] = pd.Series([string.replace(string.replace(str(x), "\"",""), "'","") for x in data[col]])
	    
	    # Better left alone, not converted to value
	    if col == "Blade_Width":
		data[col]=data[col].fillna(value =0)
		#data[col]=data[col].fillna(value =0)
		new_arr = []
		for x in data[col]:
		    if x == "<12'": repl = 0
		    elif x == "None": repl = 13.5 # calculated average for MG ProductGroup (only relevant one)
		    else: repl = float(string.replace(str(x), "'",""))
		    new_arr.append(repl)
		data[col] = new_arr
	    
	    # almost the same.. keep as is, messy value
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
		# Stick_Length : 22 : ['nan' '10\' 10"' '10\' 2"' '10\' 6"' '11\' 0"' '11\' 10"' '12\' 10"'
	    
	    # barely better.. keep as is?
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
	    
		
	# Better as value
	if col == "UsageBand":
	    data[col]=data[col].fillna(value ="")
	    d = {"Low":0, "":1,"Medium":2,"High":3}
	    data_out = map_columns(col, d, data, data_out)
        elif col == 'fiModelSeries':
	    data[col] = data[col].fillna(value = "")
	    data[col] = [str(x).strip() for x in data[col].values]
        # Create letters only model
        elif col == 'fiBaseModel':
            data[col]=data[col].fillna(value ="")
            
            data[col] = [str(x).strip() for x in data[col].values]
	    model_letters = []
	    model_l1 = []
	    model_n = []
            for x in data[col]:
                # Find letters only at beginning of name
                try:
                    m = re.search('([a-zA-Z]*)([1-9]*)',x) # only take adjacent letters
                    if m is not None:
			letters = m.group(1)
                        model_letters.append(letters)
			if len(letters) >= 1: model_l1.append(letters[0])
			else: model_l1.append("")
			if m.group(2) != "": model_n.append(float(m.group(2)))
			else: model_n.append(0)
                    else: 
			model_letters.append("")
			model_l1.append("")
			model_n.append(0)
                except TypeError: # Not sure what this is
                    model_letters.append("")
		    model_l1.append("")
		    model_n.append(0)
            #data['fiBaseModelL'] = model_letters
	    data['fiBaseModelN'] = model_n
            data['fiBaseModelL1'] = model_l1 
        # Get manufacturing ID, power min, power max, and unit based on MachineID
        elif col == "MachineID":
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
                    if power_max == "" and power_min == "":power_max, power_min = -1,-1
		    # below line not needed since most of the time both are empty together
                    elif  power_max == "" or pd.isnull(power_max): power_max = -1
                    elif power_max == 1000000: # when says e.g. "100+"
                        power_max = power_min
                    #if power_min == 0: power_min = power_max
                    #elif power_min == "":power_min = -1
                except KeyError: mfgid, power_u, power_min, power_max = 0, "", -1, -1
                mfgids.append(mfgid_temp)
		
		power_max_l.append(float(power_max))
		power_min_l.append(float(power_min))
                power_unit_l.append(power_u)
	    
            data['MfgID'] = mfgids
            data['power_max'] = power_max_l
            data['power_min'] =  power_min_l
            data['power_u'] =  power_unit_l
	    
	    print "calculating pow2er min,max means"
	    power_min_means = {}
	    power_max_means = {}
	    data["ProductSize"]=data["ProductSize"].fillna(value ="")
	    for product in ["BL", "MG", "SSL", "TEX", "TTT", "WL"]:
		#for productsize in [str(x) for x in range(7)]:
		for productsize in ["","Mini","Compact","Small","","Medium","Large / Medium","Large"]:
		    power_min_means[product + productsize] = np.mean( data[ (data["ProductGroup"] == product) &
			(data["ProductSize"] == productsize) & (data["power_min"] >= 0)]["power_min"])
		    power_max_means[product +productsize] = np.mean( data[ (data["ProductGroup"] == product) &
			(data["ProductSize"] == productsize) & (data["power_max"] >= 0)]["power_max"])
		    
	    print "done calc power means"
	    for x in xrange(num_rows):
		if data["power_min"][x] == -1:
		    product = data["ProductGroup"][x]
		    #productsize = data.xs(x)["ProductSize"]
		    productsize = str(data["ProductSize"][x])
		    data["power_min"][x] = power_min_means[product + productsize]
		    data["power_max"][x] = power_max_means[product + productsize]
	    print "power_min: done populating empty"
            #data[["power_min", "power_max"]].to_csv('power_min_out', index=True)
    
    data = data.drop(col_list,axis =1)
            
    return data, data_out
            
    #if data["UsageBand"]

def binarize_cols(col, train, test, train_fea, test_fea):
    """ Change categorical variables to binary columns"""
    lb_bm = preprocessing.LabelBinarizer()
    lb_bm.fit(np.append(train[col].values, test[col].values))
    # Create matrix of binary columns
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
    
    if testing == 1:
	# This is the to test predict
	test = train[(train["saledate"] >= datetime.datetime(2011,1,1)) & (train["saledate"] < datetime.datetime(2011,5,1))]
	test_Y = test["SalePrice"]
	train = train[(train["saledate"] < datetime.datetime(2011,1,1))]
	
	print test.shape
	train = train.reset_index(drop = True) # drops the original index
	test = test.reset_index(drop = True)
	del test["SalePrice"]
    
    train_fea = get_date_dataframe(train["saledate"])
    test_fea = get_date_dataframe(test["saledate"])
    
    train_fea = map_external_data( train_fea)
    test_fea = map_external_data(test_fea)

    print "Cleaning Columns"
    train, train_fea = clean_columns(train, train_fea)
    test, test_fea = clean_columns(test, test_fea)
    # train[["power_min", "power_max", "SalePrice", "ProductGroup", "YearMade"]].to_csv('power_min_out.csv', index=True)
    test_columns = set(test.columns)
    columns = set(train.columns)
    
    columns.remove("SalesID")
    columns.remove("SalePrice")
    columns.remove("saledate")
    for col in columns:
        # these deleted already["ProductGroupDesc", "fiProductClassDesc"]
        if col not in col_list : # REAL ONE
        #if col == "fiModelSeries":
        # col_list = ["ProductSize","UsageBand",'fiProductClassDesc'] # "ProductGroup"
        #if col == "Backhoe_Mounting": # Testing - error in the fillna "convert string to float" but why?
            #print "starting", col
            
	    # Binarize these, even if numerical
            if col  in [ 'fiBaseModelL', 'ProductGroup', 'fiSecondaryDesc', 'fiSecondaryDesc', 'state',
                        'auctioneerID', 'power_u', 'MfgID', 'Enclosure', 'SaleMonth', "SaleWkDay", "fiProductClassDesc"] :
            #if col in ['fiBaseModelL', 'ProductGroup', 'fiSecondaryDesc', 'fiSecondaryDesc', 'Enclosure', 'MfgID']:
                print "binarize", col
		
                s = np.unique(x for x in np.append(train[col].values,test[col].values))
                print col, ":", len(s), ":",s[:10]
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
                s = np.unique(x for x in np.append(train[col].values,test[col].values))
                print  col, ":", len(s), ":", s[:10]
                
                # Binarize these ones:
                if len(s) >2 and len(s) < 100 and col not in "Thumb": # in [ 'fiBaseModel']
                # try on len(s) > 3
                # don't binarize datasource (6 values), lower performance
                    #print "binarize", col
                    # would binarize BaseModel but too much memory
                    train_fea, test_fea = binarize_cols(col, train, test, train_fea, test_fea)
                
                # Just enumerate these
                else:
                #if 1:
                    print "enumerate",col
                    #Don't need below line for full data set usually
                    if test[col].dtype != np.dtype(object):
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
                        
                        train_fea[col] = train[col].map(mapping)
                        test_fea[col] = test[col].map(mapping)
                        #train_fea[col] = np.log(train_fea[col]+1)
                        #test_fea[col] = np.log(test_fea[col]+1)
                    if pd.isnull(train_fea[col]).any() or pd.isnull(test_fea[col]).any():
                        print "HAS NAN", col
                    
            else:
            # Numeric field
            
                #print col, " as number"
            #if train[col].dtype != np.dtype('object'):
                #m = np.mean([x for x in train[col] if not np.isnan(x)])
                
                # can use pd.isnull(frame).any() TRY 
                if col =="MachineID" or col == "ModelID" or col == "datasource":
                    m = 0 # value to fill
		    train_m = 0
		    test_m = 0
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

                if col in ["Stick_Length"]:
                    train[col] = train[col].replace(0,train_m)
                    test[col] = test[col].replace(0,test_m)
                else:
                    train[col] = train[col].fillna(value =train_m)
                    test[col] = test[col].fillna(value=test_m)
                print col, train_m
                
                
                # "mean is nan" 
                if np.isnan(train_m): train[col] = train[col].fillna(value =0)
                if np.isnan(test_m):test[col] = test[col].fillna(value=0)
                
                #train[col] = train[col].astype('float64')
                #test[col] = test[col].astype('float64')
                train[col] = np.log(train[col]+1)
                test[col] = np.log(test[col]+1)
                
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

    #train_fea = train_fea[train_fea["SaleYear"] <  2011] # try this out # now at the top
    #train = train[train["saledate"] <= datetime.datetime(2011,1,1)] # try this out # now at the top

    del train_fea["SaleYear"]
    del test_fea["SaleYear"]
    if testing == 1:
	return train_fea, test_fea, [x for x in train["SalePrice"]], test_Y
    else:
	return train_fea, test_fea, [x for x in train["SalePrice"]]

if testing == 0:train_fea, test_fea , train_Y  = data_to_fea()
else: train_fea, test_fea , train_Y, test_Y = data_to_fea()
print "Length of features:", train_fea.shape
#print "Length of "
train_Y = np.log(train_Y)
#train_validate_Y = np.log(train_validate["SalePrice"])

train_len = train_fea.shape[0]
print "new train length", train_len
train_len = len(train_Y)
print "new train length", train_len

if use_wiserf == 1:
    rf = WiseRF(n_estimators=50) # Doesn't work right now
elif run_ml == 1:
    print "SK Learn Running Forest"
    if testing == 1: rf = RandomForestRegressor(n_estimators=50, n_jobs=4, compute_importances = True)
    else: rf = RandomForestRegressor(n_estimators=100, n_jobs=4, compute_importances = True)
    #print "Learn Gradient Boosting" # Slower
    #rf = GradientBoostingRegressor(subsample = .67) #n_estimators = 100 by default # Cannot parallelize
    
    rf.fit(train_fea, train_Y)
    
    print "Fitting"
    
    # if testing, this is part of training set.
    predictions = rf.predict(test_fea)
    predictions = np.exp(predictions)

if testing == 0: util.write_submission("submit_rf" + comment + ".csv", predictions)

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
            
#print "oob score:", rf.oob_score_
#print "score", rf.score(train_fea, train_Y)
logger.write("\n" + comment+ "\n")
#logger.write("oob score:" +str( rf.oob_score_)+ "\n")

def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

# RMSE
print "Calculating RMSE"

train_predict = rf.predict(train_fea)
train_predict = np.exp(train_predict)
    
print "Length of train_predict", len(train_predict)
print "Length of train_validation", len(train_predict)
train_len = len(train_predict)
train_Y = np.exp(train_Y)
error_sum = 0
i =0
for i in xrange(train_len):
    error_unit = (np.log(train_Y[i]) - np.log(train_predict[i] ))**2
    error_sum += error_unit
error_sum = error_sum/float(train_len)    
rmse = np.sqrt(error_sum)
print "Train Set RMSE:", rmse
logger.write("Train Set RMSE:" + str(rmse)+ "\n")

train_fea["prediction"] = train_predict
train_fea["actual"] = train_Y
#print "Writing to csv"
#train_fea.to_csv('out/rf_train_Y.csv')
print "Starting Oob RMSE"
if testing == 1:
    train_predict = predictions
    train_Y = [ x for x in test_Y]
    train_len = len(train_predict)
    error_sum = 0
    for i in xrange(train_len):
	error_unit = (np.log(train_Y[i]) - np.log(train_predict[i] ))**2
	error_sum += error_unit
    error_sum = error_sum/float(train_len)    
    rmse = np.sqrt(error_sum)
    print "Oob Set RMSE:", rmse
    logger.write("RMSE:" + str(rmse)+ "\n")
#csv_p = csv.writer(open('out/rf_train_Y.csv','wb'))
#for i in xrange(len(train_predict)):
    #csv_p.writerow([train_predict[i], train_Y[i]])

print datetime.datetime.today()
t2 = time()
t_diff = t2-t1
print "Time Taken (seconds):", round(t_diff,0)

logger.write("Time:" + str(round(t_diff,0))+ "\n")
