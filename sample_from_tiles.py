import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd
#navigate to folder containing points cut to raster shape
allpoints = glob.glob('data/SGU/sgu_slu_lev2/split_csv/clipped_2/*.shp')

for point in allpoints:
    #read shape file into geodataframe
    pts = gpd.read_file(point)
    #empty files have a size of 100
    if os.stat(point).st_size == 100:
        print('File is empty')
        continue
    else:
        print('File is not empty')
        #split path to obtain the tile number
        pts_name = point.split("/")[-1].split('.')[0]
        print('reading point tile number :', pts_name)
        coords = [(x,y) for x, y in zip(pts.POINT_X, pts.POINT_Y)]
        #open raster stack tile using point shape file name
        src=rio.open('data/SGU/orginal_fran_wl/CompositeBands/{:s}.tif'.format(pts_name))
        proper_col_name={1:"DEM", 2:"EAS1ha", 3:"EAS10ha",4:"DI2m",
        5:"CVA",6: "SDFS",7: "DFME",8: "Rugged",9:"NMD",11:"HKDepth", 12:"SoilDepth",13:"LandAge",
        14:"MSRM",17:"MED"}
        for i in range (1, src.count+1):
            if i in [10, 15, 16]:#skip band 10, 15, 16 which stand for soil, x,y 
                continue
            pts[proper_col_name[i]] = [x[i-1] for x in src.sample(coords)]#extract value and attach to column, extraction is done for all bands, and x[i-1] ensures it only attaches the corresponding column value 
            print(proper_col_name[i], ' attached!')
            print(pts)
        pts.to_csv('data/SGU/sgu_slu_lev2/split_csv/output_karttyp2/output_{:s}.csv'.format(pts_name))#save back to a csv
        print('output {:s} , csv saved!'.format(pts_name))
