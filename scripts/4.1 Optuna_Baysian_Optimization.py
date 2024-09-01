import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report,cohen_kappa_score,f1_score,matthews_corrcoef
import matplotlib.pyplot as plt
import os
import random
from sklearn.preprocessing import LabelEncoder 
import optuna
from optuna import Trial, visualization
import plotly
from optuna.samplers import TPESampler
# import data
df_train_sfsi = pd.read_csv('/workspace/data/SGU/SFSI/SFSI_cleaned.csv',sep = ',', decimal = '.', index_col=False)
df_train_sfsi
#clean up dataset 
var_columns_sfsi = [c for c in df_train_sfsi.columns if c not in ['Unnamed: 0','OBJECTID', 'MarkInvent', 'Taxar', 'TraktNr', 'PalslagNr', 'DelytaNr',
       'SampleType', 'H_form', 'H_thick', 'E_thick', 'Bs', 'ParentMate',
       'Texture', 'ParentMa_1', 'Texture_M6', 'SoilDepth', 'SoilTypeWR',
       'CultSoilTy', 'Disturbed', 'B_lowerLim', 'H_sampleTa', 'M_SampleTa',
       'Landuse', 'PitDist', 'PitDirecti', 'SoilMoistu', 'Humificati',
       'Humifica_1', 'Ditch', 'Pit_Ycoord', 'Pit_Xcoord', 'East_coord',
       'North_coor', 'x_point', 'y_point', 'geometry', 'Tabell3_4','GENERAL_TX', 'Texture_TX', 'PaMe_TX']]
df_train_sfsi= df_train_sfsi[var_columns_sfsi]

#stratify random sample 20% of data for testing
sfsi_to_test = df_train_sfsi.groupby('PaMa/Texture', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=123))
#drop the 20% reserved Test data from sfsi dataframe
df_train_sfsi = df_train_sfsi.drop(sfsi_to_test.index)
# slice the remaining dataset into x and y foir features and labels
x = df_train_sfsi.loc[:,[c for c in df_train_sfsi.columns if c not in ['PaMa/Texture']]]
y = df_train_sfsi.loc[:,'PaMa/Texture']
#define objective for bayesian optimization 
def objective(trial):

    # Define the hyperparameters to optimize
    params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 26,
        'tree_method': 'gpu_hist',
         # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 0., 50.0),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 0., 10.0),
        # defines booster
        "booster": trial.suggest_categorical("booster", ["gbtree","dart"]),
        # maximum depth of the tree, signifies complexity of the tree.
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'eta': trial.suggest_float('eta', 0.001, 0.3),
        # defines how selective algorithm is.
        'gamma': trial.suggest_float('gamma', 0,10 ),
         # sampling according to each tree.
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.4,0.9),
        # sampling ratio for training data.
        'subsample':trial.suggest_float('subsample',0.4,0.9),
        # minimum child weight, larger the term more conservative the tree.
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 10)}
    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=False)
        params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=False)

    
    #Perform stratified k-fold cross-validation on the XGBoost model
    scv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score_mcc = []
    #score_f1 = []
    #score_cohenskappa = []
    for train_idx, val_idx in scv.split(x, y):
        X_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.fit_transform(y_val)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
    # Train the XGBoost model
        model = xgb.train(params, dtrain, early_stopping_rounds=20, evals=[(dval, 'eval')], verbose_eval=True)

    # Make predictions on the testing set
        y_val_pred = model.predict(dval)

    # Calculate the e´valuation metrics
        #cohen_kappa_score_test = cohen_kappa_score(y_val, y_val_pred)
        #f1_score_test = f1_score(y_val, y_val_pred, average='macro')
        #matthews_corrcoef_test = matthews_corrcoef(y_val, y_val_pred)
        score_mcc.append(matthews_corrcoef(y_val, y_val_pred))
        #score_f1.append(f1_score(y_val, y_val_pred, average='macro'))
        #score_cohenskappa.append(cohen_kappa_score(y_val, y_val_pred))

    return np.mean(score_mcc)#, np.mean(score_f1), np.mean(score_cohenskappa)
  
# calling the optuna study
study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(objective, n_trials= 300, show_progress_bar = True)


# Get the best hyperparameters and the corresponding scores
best_params = study.best_params
best_score_mcc = study.best_value

# Save the results to a dataframe and a CSV file
df = study.trials_dataframe()
df.to_csv('/workspace/data/SGU/SFSI/optuna_sfsi_results.csv', index=False)

#visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_slice(study)
