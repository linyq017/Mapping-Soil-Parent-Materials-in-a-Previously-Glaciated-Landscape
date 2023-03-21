
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report,cohen_kappa_score,f1_score,matthews_corrcoef
from sklearn.preprocessing import LabelEncoder 
import random


#import trainingdata
os.chdir('/Volumes/Extension_100TB/Lin/data/SGU/SFSI/')
df_train_sfsi = pd.read_csv('extracted_SFSI - Copy.csv',sep = ';', decimal = ',', encoding='latin-1')
print(df_train_sfsi)
#slice useful columns out from the entire dataset
var_columns_sfsi = [c for c in df_train_sfsi.columns if c not in ['SampleType', 'OBJECTID_1','FID',
 'OBJECTID','H_form', 'H_thick','E_thick','Bs','SoilTypeWR','SoilDepth','id','Pit_xcoord', 'Pit_ycoord', 
 'index','MarkInvent', 'Taxar', 'TraktNr', 'PalslagNr', 'DelytaNr','DOWNSLOPEI','x_point', 'y_point','SFSI_field_TX',
       'SampleType','ParentMate','Texture', 'ParentMa_1', 'Texture_M6','CultSoilTy', 'Disturbed', 'B_lowerLim', 'H_sampleTa', 'M_SampleTa',
       'Landuse', 'PitDist', 'PitDirecti', 'SoilMoistu', 'Humificati','ParentMa_2','NMD',
       'Humifica_1', 'Ditch', 'East_coord', 'North_coor','geometry','SFSI_field','ParentMate_TX', 'Texture_TX']]
df_train_sfsi = df_train_sfsi[var_columns_sfsi]
#define kfold
kfold = 5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#change datatype because original datatype is object
cols_to_change = [c for c in df_train_sfsi if c not in 'GENERAL_TX']
df_train_sfsi[cols_to_change] = df_train_sfsi[cols_to_change].astype('float')
#slice x (training features) and y (labels)
x = df_train_sfsi.loc[:,[c for c in df_train_sfsi.columns if c not in ['GENERAL_TX']]]
y = df_train_sfsi.loc[:,'GENERAL_TX']

# A parameter grid for XGBoost
learning_rate_list = [0.02, 0.05, 0.1, 0.2, 0.3]
max_depth_list = [6,8]
num_boost_round_list = [100]
colsample_bytree_list = [0.7,0.5]
min_child_weight_list =[1,5, 10]
alpha_list = [0]
lambda_list = [1, 10, 50]


#display cross validation result with corresponding hyperparameters
res_dict = {}#create a dictionary so results can append
res_dict["learning rate"]=[]
res_dict["max depth"] = []
res_dict["num_boost_round"] = []
res_dict["colsample_bytree"] = []
res_dict["min_child_weight"] = []
res_dict["alpha"] = []
res_dict["lambda"] = []
res_dict["training_accuracy"] = []
res_dict["testing_accuracy"] = []
res_dict["cohen_kappa_score_test"] = []
res_dict["f1_score_test"] = []
res_dict["matthews_corrcoef_test"] = []
for i in learning_rate_list:
    for j in max_depth_list:
        for l in num_boost_round_list:
            for h in colsample_bytree_list:
                for m in min_child_weight_list:
                    for n in alpha_list:
                        for o in lambda_list:

                            params = {'learning_rate': i,
                                'max_depth' : j,
                                'colsample_bytree': h, 'min_child_weight': m,
                                'lambda': o, 'alpha':n,
                                'objective':'multi:softmax', 'subsample':0.5,
                                'num_class': 6, 'verbosity':1,  "max_cat_to_onehot":6, 'eval_metric':["mlogloss","merror"]}
            #Cross Validation
                            #define x y for train and validation fold from stratified kfold
                            ii = 0 #CV
                            for train_idx, val_idx in skf.split(x, y):
                                ii +=1 #ii = i+1 
                                X_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
                                X_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
                                le = LabelEncoder()
                                y_train = le.fit_transform(y_train)
                                y_val = le.fit_transform(y_val)
                                #print each CV hyperparameters when code is running                          
                                print("CV: {:s}; num_boost_round: {:s};  max_depth: {:s};  learning_rate: {:s}; colsample_bytree: {:s}; min_child_weight: {:s}; alpha: {:s}; lambda: {:s}".format(str(ii),str(l),str(j),str(i),str(h),str(m),str(n),str(o)))
                
                                #convert data to dmatrix data structure to utilize xgboost fast processing
                                dtrain = xgb.DMatrix(X_train, label=y_train)
                                dval = xgb.DMatrix(X_val, label=y_val)
                                evals_result = {}
                                model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10, evals=[(dval, 'eval')], verbose_eval=True,evals_result=evals_result)
                                print( model.best_score)                                 
                                # prediction result on train and val set in order to compute other metrics 
                                y_train_pred = model.predict(dtrain)
                                y_val_pred = model.predict(dval)
                                #append results to results d
                                res_dict["learning rate"].append(i)
                                res_dict["max depth"].append(j)
                                res_dict["num_boost_round"].append(l)
                                res_dict["colsample_bytree"].append(h)
                                res_dict["min_child_weight"].append(m)
                                res_dict["alpha"].append(n)
                                res_dict["lambda"].append(o)
                                train_accuracy = accuracy_score(y_train, y_train_pred)
                                res_dict["training_accuracy"].append(train_accuracy)
                                #print("training accuracy:", train_accuracy)
                                #print(classification_report(y_train, y_train_pred))
                                test_accuracy = accuracy_score(y_val, y_val_pred)
                                res_dict["testing_accuracy"].append(test_accuracy)
                                #print("testing accuracy:", test_accuracy)
                                #                                     #print(classification_report(y_test,y_test_pred))
                                cohen_kappa_score_test = cohen_kappa_score(y_val, y_val_pred)
                                res_dict['cohen_kappa_score_test'].append(cohen_kappa_score_test)
                                f1_score_test = f1_score(y_val, y_val_pred, average='macro')
                                res_dict['f1_score_test'].append(f1_score_test)
                                matthews_corrcoef_test = matthews_corrcoef(y_val, y_val_pred)
                                res_dict['matthews_corrcoef_test'].append(matthews_corrcoef_test)
print(res_dict)

#convert result from dictionary to dataframe and sort by values
res_df = pd.DataFrame.from_dict(res_dict)
res_df.sort_values(by=['matthews_corrcoef_test'])
res_df.to_csv('/Volumes/Extension_100TB/Lin/data/SGU/SFSI/hypertune_SFSIcorrect.csv',sep = ';', decimal = '.', encoding='latin-1')
print('results saved to csv!')
