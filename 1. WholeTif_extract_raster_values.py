import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd

# point vector file
pointfile = gpd.read_file('/workspace/data/SGU/SFSI/project_shapefile/sfsi3.shp')
coords = [(x,y) for x, y in zip(pointfile.newx_point, pointfile.newy_point)]
# folder containing single big tifs where all points fall into
tif_folder = '/workspace/data/wbt/newtifs'
for tif_file in os.listdir(tif_folder):
    if tif_file.endswith('.tif'):
        tif_path = os.path.join(tif_folder, tif_file)
        print("Reading file:", tif_file)
        with rio.open(tif_path) as src:
            # Read the raster values at the point locations
            values = [x[0] for x in src.sample(coords)]
            # Add the raster values to the GeoDataFrame
            pointfile[tif_file] = values

print("Extraction completed.")
            
