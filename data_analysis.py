import pandas as pd
import numpy as np


def reading_the_files():
    
    """reading the input files for the project"""
    # reading the input file
    data = pd.read_csv(r"C:\Users\Chandu Jagarlamundi\Desktop\Thesis_Wind data\Data Wind_extern\data_eng.csv")
    # reading the anlagestatus file, to get the details regarding anlage status
    status_data = pd.read_excel(r"C:\Users\Chandu Jagarlamundi\Desktop\Thesis_Wind data\Data Wind_extern\system_status.xlsx")      
    return data, status_data


def preprocessing(data):    
    # deleting the columns which have lots of missing values
    columns_data = data.columns
    for i in columns_data:
        if data[i].isna().sum()>(len(data)/2):
            print("The number of missing values in the column {} are {} ".format(i,data[i].isna().sum()))
            data.drop([i], axis = 1, inplace = True)
    columns_data = data.columns
    
    # deleting the columns which have single values
    for i in columns_data:
        if len(data[i].value_counts())<2:
            print("{} is eliminated".format(i))
            data.drop([i], axis = 1, inplace = True)
            
    data.Equipment = data.Equipment[data.Equipment!='Anlage']
    
    # dropping the missing rows
    data.dropna(axis = 0, inplace=True)
    data.drop(['Date(Remote)', 'Time(Remote)', 'Date(Server)', 'Time(Server)', "operating_state"], axis=1, inplace=True)
    data.Equipment = pd.Categorical(data.Equipment, categories=data.Equipment.unique()).codes
    # one hot encoding the Equipment feature
    data = pd.concat([data, pd.get_dummies(data.Equipment)], axis=1)
    data.drop(['Equipment'], axis=1, inplace=True)
    return data


def splitting_valid_float(data, status_data):
    """ splitting the data into valid, invalid datasets and 
    then mapping the valid dataset with the status text"""
    # getting the true values of systemstatus
    common_status = np.intersect1d(status_data['Status_Number'], data['system_status'])
    # getting the valid data
    data_valid = data[data.system_status.isin(common_status)]
    data_float = data[~data.system_status.isin(common_status)]
    # converting the required data from the status_data into a dictionary    
    return data_valid, data_float


