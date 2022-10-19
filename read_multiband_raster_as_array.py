from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import sys
import xgboost as xgb
from osgeo import gdal_array
from osgeo import osr
import rasterio as rio  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#open a raster tile
#ds = gdal.Open('data/SGU/orginal_fran_wl/CompositeBands/14.tif')

#loop over all tiles
allfiles = glob.glob('data/SGU/*.tif')

for tile in allfiles:
    tile_title=tile.split("/")[-1]#split path with '/' and take the last element

    ds = gdal.Open(tile)
    print('bands =', ds.RasterCount)#raster has 16 bands

#bands and corresponding numbers
# names(img)[1]<-paste("DEM") 
# names(img)[2]<-paste("EAS1ha")
# names(img)[3]<-paste("EAS10ha")
# names(img)[4]<-paste("DI2m")
# names(img)[5]<-paste("CVA")
# names(img)[6]<-paste("SDFS")
# names(img)[7]<-paste("DFME")
# names(img)[8]<-paste("Rugged")
# names(img)[9]<-paste("NMD")
# names(img)[10]<-paste("soil")
# names(img)[11]<-paste("HKDepth")
# names(img)[12]<-paste("SoilDepth")
# names(img)[13]<-paste("LandAge")
# names(img)[14]<-paste("MSRM")
# names(img)[15]<-paste("POINT_X")
# names(img)[16]<-paste("POINT_Y")
# names(img)[17]<-paste("MED")
# 

#########################READ RASTER INTO NUMPY ARRAY###########################
#read each band into a numpy array, drop x, y and soil 
    bands_list=[]
    for i in range (1, ds.RasterCount+1):
        if i in [10, 15, 16]:
            continue
        band = ds.GetRasterBand(i)#get the  band and read as numpy array
        bands_list.append(band.ReadAsArray())

    all_data = np.array(bands_list)#make list into array
#reshaping numpy array into a table with 14 columns and 1250*1250 rows (size of the raster tiles)
    all_data=all_data.reshape(14,1250*1250).T
    print(all_data.shape)

#make np array a dataframe
    df_data=pd.DataFrame(all_data,columns=["DEM","EAS1ha","EAS10ha","DI2m",
    "CVA", "SDFS", "DFME", "Rugged","NMD","HKDepth","SoilDepth","LandAge",
    "MSRM","MED"])
#prediction data has to have the same column order as in the model
    cols=['CVA', 'SDFS', 'Rugged', 'DFME', 'MSRM', 'MED', 'EAS1ha', 'EAS10ha', 'DEM', 'DI2m', 'LandAge', 'SoilDepth', 'HKDepth', 'NMD']
    df_data=df_data[cols]
    print(df_data.head(20))
#print(all_data[0])

#dmatrix format for xgoost model
    all_data_d = xgb.DMatrix(df_data) 
    print('dataframe turned into dmatrix!')
#pd.DataFrame(all_data).to_csv("data/SGU/rasterdata.csv")
#print('finished saving csv!')



##########################XGBOOST MODEL######################
#import trainingdata
    df_train = pd.read_csv('data/SGU/orginal_fran_wl/TrainingData/TrainingData.txt', sep = '\t', decimal = ',')
#slice x training and y predicting variables
    var_columns = [c for c in df_train.columns if c not in ['Process', 'Proesskod','soil', 'POINT_X', 'POINT_Y']]
    x = df_train.loc[:,var_columns]
    y = df_train.loc[:,'Proesskod']

#encode y label to start from 0 xgboost requirement 
    le = LabelEncoder()
    y_new = le.fit_transform(y)

#split data into 70% training and 30% testing 
    x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size = 0.3, random_state=42, stratify = y)
#make DMatrix for training and test
    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
#set parameters for xgboost model
    param = {'learning_rate': 0.3, 'max_depth' : 12, 'objective':'multi:softmax', 'subsample':0.5, 'n_estimators':1000, 'lambda': 5,
                             'num_class': 7, 'verbosity':1, 'colsample_bytree':0.5, 'min_child_weight':1}
    epochs = 10
#training 
    model = xgb.train(param, train, epochs)
#prediction on raster tile
    pred = model.predict(all_data_d)
    print(len(pred))
    print('prediction made!')
###############################SAVE DATA BACK TO RASTER##########################
#reshape numpy array into 1250x1250
    pred = pred.reshape(1250,1250)
    print(pred)

####save numpy back to raster
    with rio.open(tile) as src:
        ras_data = src.read()
        ras_meta = src.profile
        ras_meta.update(count=1)


    with rio.open('data/'+'output_'+tile_title, 'w', **ras_meta) as dst:
        dst.write(pred, 1)
        print('tile {:s} , raster saved!'.format(tile_title))



