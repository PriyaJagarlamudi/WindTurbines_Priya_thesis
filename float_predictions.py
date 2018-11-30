# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:02:57 2018

@author: Chandu Jagarlamundi
"""

from data_analysis import reading_the_files, preprocessing, splitting_valid_float
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score



def split_data_valid(data_valid):
    # obtaining the fraction of the data
    data_sample= data_valid.sample(frac=0.2).reset_index(drop=True)
    target_sample = data_sample['system_status']
    #####an error here###################
    data_sample.drop(['system_status'], axis=1, inplace=True)
    system_stats = pd.factorize(target_sample)
    target_sample = system_stats[0]
    train_x, val_x, train_y, val_y = train_test_split(
            data_sample, target_sample, test_size=0.33, random_state=42)
    return train_x, val_x, train_y, val_y 


def processing_data_float(data_float):
    target_float = pd.Categorical(data_float['system_status'])
    data_float.drop(['system_status'], axis=1, inplace=True)    
    return data_float, target_float 


def classifier_float(train_x, val_x, train_y, val_y , data_float, target_float):
    """applying the random forest for the set of data"""
    choice = input('rf or dt')
    if choice == 'rf':    
        classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
        classifier_rf.fit(train_x, train_y)
        pred_y_rf = classifier_rf.predict(val_x)
        acc_rf_valid = accuracy_score(val_y, pred_y_rf)
        print('The accuracy of the random forest is: {}'.format(acc_rf_valid))
        pred_float_rf = classifier_rf.predict(data_float)
        return pred_float_rf
    elif choice == 'dt':
        classifier_dt = DecisionTreeClassifier(random_state=0)
        #cv = cross_val_score(classifier_dt , data_sample, target_sample, cv=10)
        classifier_dt.fit(train_x, train_y)
        pred_y_rf = classifier_dt.predict(val_x)
        print("The accuracy of decision tree is: ", accuracy_score(pred_y_rf, val_y))
        feat_importances = pd.Series(classifier_dt.feature_importances_, index=val_x.columns)
        feat_importances.nlargest(25).plot(kind='barh')
        pred_float_dt = classifier_dt.predict(data_float)
        return pred_float_dt
        


def writing_file(pred_float_rf, target_float):
    file = open('pred_float.txt', 'w')
    for i in pred_float_rf:
        file.write(str(i)+'\n')
    file.close()
    file = open('target_float.txt', 'w')
    for i in target_float:
        file.write(str(i)+'\n')
    file.close()
    
def whole_data_test(data):
    # obtaining the fraction of the data
    data_sample= data.sample(frac=0.2).reset_index(drop=True)
    target_sample = data_sample['system_status']
    data_sample.drop(['system_status'], axis=1, inplace=True)
    system_stats = pd.factorize(target_sample)
    target_sample = system_stats[0]
    train_x, val_x, train_y, val_y = train_test_split(
            data_sample, target_sample, test_size=0.33, random_state=42)
    return train_x, val_x, train_y, val_y 

files = reading_the_files()
data = files[0]
status_data = files[1]
data = preprocessing(data)
data_split = splitting_valid_float(data, status_data)
data_valid = data_split[0]
data_float = data_split[1]
data_whole = whole_data_test(data_valid)
train_x, val_x, train_y, val_y  = data_whole[0], data_whole[1], data_whole[2], data_whole[3]
float_data = processing_data_float(data_float)
data_float = float_data[0]
target_float = float_data[1]
pred_float = classifier_float(train_x, val_x, train_y, val_y, data_float, target_float)
#f1_score = f1_score(target_float, pred_float, average='macro')
print(pred_float)