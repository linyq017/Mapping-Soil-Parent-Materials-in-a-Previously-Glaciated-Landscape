import numpy as np
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import glob
from rasterio.coords import BoundingBox
from shapely.geometry import box
#all nultiband raster stacks used to clip point file
allfiles = glob.glob('/workspace/data/SGU/orginal_fran_wl/CompositeBands/*.tif')
#point vector file
pointfile = gpd.read_file('/workspace/data/SGU/SFSI/project_shapefile/sfsi3.shp')

for file in allfiles:
    print('current file:', file)
    #split names and take just the number
    name = file.split('.')[0].split('/')[-1]
    #rasterio bounds gives a format of left =, bottom =, right = , top =
    #bounds = rio.open(file).bounds
    bbox = rio.open(file).bounds
    bounds = BoundingBox(left=bbox.left, bottom=bbox.bottom, right=bbox.right, top=bbox.top)
    poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    #pointfile = pointfile.to_crs(file.crs)
    #bound_list = [bounds[0],bounds[1],bounds[2],bounds[3]]
    print('bounding box:', bounds)
    #gpd
    clipped = gpd.clip(pointfile,poly)
    try:
        clipped.to_file('/workspace/data/SGU/SFSI/SFSI/new_clips/{:s}.shp'.format(name))
    except:
        continue

