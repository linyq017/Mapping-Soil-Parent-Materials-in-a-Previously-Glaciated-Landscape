import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob

#loop over all csv chunks
allfiles = glob.glob('data/SGU/till_slu_lev1_csv/CenterPoint2_chunks/integer_chunks/*.csv')
print(allfiles)
for file in allfiles:
    pts = pd.read_csv(file, encoding='latin-1')
    #split file to optain file name for ease of assigning names to new csv outputs
    file_name=file.split("/")[-1]#take the last element split with '/'s. e.g. chunk1 from 'data/SGU/till_slu_lev1_csv/CenterPoint2_chunks/integer_chunks/chunk0.csv'
    chunk_num=file_name.split('.')[0]#split with '.', e.g. chunk35.csv, so it takes the first element before '.'
    #slice dataframe (in this case not necessary because csv only has XY columns)
    pts = pts[['POINT_X', 'POINT_Y']]
    pts.index = range(len(pts))
    print(pts)
    #list of XY coordinates
    coords = [(x,y) for x, y in zip(pts.POINT_X, pts.POINT_Y)]
#print(coords)
    proper_col_name={1:"DEM", 2:"EAS1ha", 3:"EAS10ha",4:"DI2m",
        5:"CVA",6: "SDFS",7: "DFME",8: "Rugged",9:"NMD",11:"HKDepth", 12:"SoilDepth",13:"LandAge",
        14:"MSRM",17:"MED"}
    src = rio.open('data/SGU/orginal_fran_wl/tif/merged2.vrt')

#go through bands and sample, attach new column to dataframe
    for i in range (1, src.count+1):
        if i in [10, 15, 16]:#skip band 10, 15, 16 which stand for soil, x,y 
            continue
        pts[proper_col_name[i]] = [x[i-1] for x in src.sample(coords)]#extract value and attach to column, extraction is done for all bands, and x[i-1] ensures it only attaches the corresponding column value 
        print(proper_col_name[i], ' attached!')

#inspect results and save output
        print(pts)#Print every column out after attachment
    pts.to_csv('data/SGU/till_slu_lev1_csv/CenterPoint2_chunks/output/'+'output_'+ file_name)#save back to a csv
    print('output {:s} , csv saved!'.format(chunk_num))
