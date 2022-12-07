import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report,cohen_kappa_score,f1_score,matthews_corrcoef
from sklearn.preprocessing import LabelEncoder 
import random


#import trainingdata
os.chdir('data/SGU/sgu_slu_lev2/split_csv/extracted_values_to_csv')
df_train = pd.read_csv('allpoints_cleaned.csv', decimal = '.', encoding='latin-1')


#slice x training and y predicting variables
var_columns = [c for c in df_train.columns if c not in ["index","JG2","JG2_TX","KARTERING","KARTTYP","SHAPE_STAr","SHAPE_STLe","TYPE","JG2_AI","JG2_AI_TX","JG2_AI_TX_EN",'Process', 'Proesskod','soil']]
x = df_train.loc[:,var_columns]
y = df_train.loc[:,"JG2_AI"]

#encode y label to start from 0 xgboost requirement 
le = LabelEncoder()
y_new = le.fit_transform(y)

#hyperparameter tuning with Gridsearchcv
learning_rate_list = [0.05, 0.1, 0.2, 0.3]
max_depth_list = [9,12,15]
num_boost_round_list = [100,150,200,250,300,350,400,450,500]
epochs = 10
CV=5

#display cross validation result with corresponding hyperparameters
res_dict = {}
res_dict["learning rate"]=[]
res_dict["max depth"] = []
res_dict["num_boost_round"] = []
res_dict["CV"] = []
res_dict["training_accuracy"] = []
res_dict["testing_accuracy"] = []
res_dict["cohen_kappa_score_test"] = []
res_dict["f1_score_test"] = []
res_dict["matthews_corrcoef_test"] = []
for i in learning_rate_list:
    for j in max_depth_list:
        for l in num_boost_round_list:
            param = {'learning_rate': i,
                     'max_depth' : j,
                     'objective':'multi:softmax', 'subsample':0.5,
                     'num_class': 39, 'verbosity':1, 'colsample_bytree':0.5, 'min_child_weight':1, "max_cat_to_onehot": 39}
            #Cross Validation
            for cross in range(CV):
                x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size = 0.3, 
                                                                random_state=random.randint(0,1000000), stratify = y)
                #x_train.head(), x_test.shape,y_train.shape, y_test.shape
                print("CV "+str(cross+1)+"/"+str(CV))
                #print("num_boost_round: "+str(l)+";  max_depth: "+str(j)+";  learning_rate: "+str(i))
                print("num_boost_round: {:s};  max_depth: {:s};  learning_rate: {:s}".format(str(l),str(j),str(i)))
                
                train = xgb.DMatrix(x_train, label=y_train)
                test = xgb.DMatrix(x_test, label=y_test)
                model = xgb.train(param, train, num_boost_round=l)
                y_train_pred = model.predict(train)
                y_test_pred = model.predict(test)
                
                
            
                
                res_dict["learning rate"].append(i)
                res_dict["max depth"].append(j)
                res_dict["num_boost_round"].append(l)
                res_dict["CV"].append(cross+1)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                res_dict["training_accuracy"].append(train_accuracy)
                #print("training accuracy:", train_accuracy)
                #print(classification_report(y_train, y_train_pred))
                test_accuracy = accuracy_score(y_test, y_test_pred)
                res_dict["testing_accuracy"].append(test_accuracy)
                #print("testing accuracy:", test_accuracy)
                #print(classification_report(y_test,y_test_pred))
                cohen_kappa_score_test = cohen_kappa_score(y_test, y_test_pred)
                res_dict['cohen_kappa_score_test'].append(cohen_kappa_score_test)
                f1_score_test = f1_score(y_test, y_test_pred, average='macro')
                res_dict['f1_score_test'].append(f1_score_test)
                matthews_corrcoef_test = matthews_corrcoef(y_test, y_test_pred)
                res_dict['matthews_corrcoef_test'].append(matthews_corrcoef_test)
print(res_dict)

#convert result from dictionary to dataframe and sort by values
res_df = pd.DataFrame.from_dict(res_dict)
res_df.sort_values(by=['matthews_corrcoef_test'])
res_df.to_csv('hypertune_metrics2.csv',sep = ';', decimal = '.', encoding='latin-1')
print('results saved to csv!')
