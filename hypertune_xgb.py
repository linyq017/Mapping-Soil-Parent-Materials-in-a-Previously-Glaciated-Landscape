
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder 
import random

#import trainingdata
os.chdir('data')
df_train = pd.read_csv('TrainingData.txt', sep = '\t', decimal = ',')


#slice x training and y predicting variables
var_columns = [c for c in df_train.columns if c not in ['Process', 'Proesskod', 'soil']]
x = df_train.loc[:,var_columns]
y = df_train.loc[:,'Proesskod']

#encode y label to start from 0 xgboost requirement 
le = LabelEncoder()
y_new = le.fit_transform(y)

#hyperparameter tuning with Gridsearchcv
learning_rate_list = [0.2, 0.3, 0.4]
max_depth_list = [9, 12,15]
n_estimators_list = [500, 1000, 1500]#[500, 1000, 1500]
epochs = 10
CV=5

#display cross validation result with corresponding hyperparameters
res_dict = {}
res_dict["learning rate"]=[]
res_dict["max depth"] = []
res_dict["n_estimators"] = []
res_dict["CV"] = []
res_dict["training_accuracy"] = []
res_dict["testing_accuracy"] = []
for i in learning_rate_list:
    for j in max_depth_list:
        for l in n_estimators_list:
            param = {'learning_rate': i,
                     'max_depth' : j,
                     'n_estimators' : l,
                     'objective':'multi:softmax', 'subsample':0.5,
                     'num_class': 7, 'verbosity':1, 'colsample_bytree':0.5, 'min_child_weight':1, "max_cat_to_onehot": 26}
            #Cross Validation
            for cross in range(CV):
                x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size = 0.3, 
                                                                random_state=random.randint(0,1000000), stratify = y)
                #x_train.head(), x_test.shape,y_train.shape, y_test.shape
                train = xgb.DMatrix(x_train, label=y_train)
                test = xgb.DMatrix(x_test, label=y_test)
            
                model = xgb.train(param, train, epochs)
                y_train_pred = model.predict(train)
                y_test_pred = model.predict(test)
                
                
                #print("CV "+str(cross+1)+"/"+str(CV))
                #print("n_estimators: "+str(l)+";  max_depth: "+str(j)+";  learning_rate: "+str(i))
                train_accuracy = accuracy_score(y_train, y_train_pred)
                res_dict["learning rate"].append(i)
                res_dict["max depth"].append(j)
                res_dict["n_estimators"].append(l)
                res_dict["CV"].append(cross+1)
                res_dict["training_accuracy"].append(train_accuracy)
                #print("training accuracy:", train_accuracy)
                #print(classification_report(y_train, y_train_pred))
                test_accuracy = accuracy_score(y_test, y_test_pred)
                res_dict["testing_accuracy"].append(test_accuracy)
                #print("testing accuracy:", test_accuracy)
                #print(classification_report(y_test,y_test_pred))
print(res_dict)
