from dateutil.parser import parse
import pandas as pd
import os
import csv

testing = 1
def get_paths():
    """
    Redefine data_path and submissions_path here to run the benchmarks on your machine
    """
    #os.environ["DataPath"] = "/home/ling/bluebook/"
    #data_path = os.path.join(os.environ["DataPath"], "FastIron", "Release")
    data_path = ""
    #submission_path = os.path.join(os.environ["DataPath"], "FastIron", "Submissions")
    submission_path = ""
    return data_path, submission_path

def get_train_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()
    if testing == 1:
        train = pd.read_csv(os.path.join(data_path, "Train.csv"),
        converters={"saledate": parse}   , na_values =["None or Unspecified"]    ,
          skiprows = [i for i in xrange(401127) if i % 10 != 0] )
    else:
        train = pd.read_csv(os.path.join(data_path, "Train.csv"),
        converters={"saledate": parse} , na_values =["None or Unspecified"]  ) #    ,
         # skiprows = [i for i in xrange(401127) if i % 100 != 0] )
        
    # For now treat None as blank
    print "Length of Train csv", len(train)
    return train 

def get_test_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()

    test = pd.read_csv(os.path.join(data_path, "Valid.csv"),
        converters={"saledate": parse},na_values =["None or Unspecified"] ) # ,
        #skiprows = [i for i in xrange(11574) if i % 1000 != 0] )
        #skiprows = range(10000,11574))
        
    return test 

def get_appendix(data_path = None):
    #if data_path is None:
        #data_path, submission_path = get_paths()
    appendix = {}
    csv_file_object = csv.reader(open('Machine_Appendix.csv', 'rb')) #Load in the training csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    for row in csv_file_object: #Skip through each row in the csv file
        appendix[int(row[0])] = [row[11], row[13],row[14],row[15]] # Set key Machine ID to Manufacturer ID
    return appendix

def get_external_data():
    ext_dict = {}
    # used to use construction_spending.csv
    ext_data = pd.read_csv('external_data/external_data.csv', index_col = 'observation_date')
    return ext_data
    # Old way
    #csv_file_object = csv.reader(open('external_data/construction_spending.csv','rU'))
    #header = csv_file_object.next() #Skip the fist line as it is a header
    #for row in csv_file_object: #Skip through each row in the csv file
        #ext_dict[row[0]] = [row[1], row[2], int(row[1])] # Set key Machine ID to Manufacturer ID
        #print row[0], row[1]
    #return ext_dict

def get_train_test_df(data_path = None):
    return get_train_df(data_path), get_test_df(data_path)

def write_submission(submission_name, predictions, submission_path=None):
    if submission_path is None:
        data_path, submission_path = get_paths()
    
    test = get_test_df()    
    test = test.join(pd.DataFrame({"SalePrice": predictions}))

    test[["SalesID", "SalePrice"]].to_csv(os.path.join(submission_path,
        submission_name), index=False)