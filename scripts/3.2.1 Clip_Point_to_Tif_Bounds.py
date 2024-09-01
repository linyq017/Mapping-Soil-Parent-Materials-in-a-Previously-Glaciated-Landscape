import numpy as np
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import glob
from rasterio.coords import BoundingBox
from shapely.geometry import box

"""
This script clips a point shapefile based on the bounding boxes of multiple raster files 
and saves the clipped points to new shapefiles.

Usage:
- Use this script when you have a set of raster files and you want to extract points 
  from a shapefile that fall within the spatial extent of each raster.
- The script outputs a new shapefile for each raster, containing only the points that 
  lie within the bounds of that raster.
"""

# Gather all the multiband raster stack files to be used for clipping the point file
allfiles = glob.glob('/workspace/data/SGU/orginal_fran_wl/CompositeBands/*.tif')

# Load the point vector file (shapefile) using GeoPandas
pointfile = gpd.read_file('/workspace/data/SGU/SFSI/project_shapefile/sfsi3.shp')

# Iterate over each raster file
for file in allfiles:
    print('Current file:', file)
    
    # Extract the base name (without extension) from the file path
    name = os.path.splitext(os.path.basename(file))[0]
    
    # Open the raster file using Rasterio to get its bounding box
    with rio.open(file) as src:
        bbox = src.bounds
    
    # Convert the bounding box into a Shapely Polygon
    bounds = BoundingBox(left=bbox.left, bottom=bbox.bottom, right=bbox.right, top=bbox.top)
    poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    print('Bounding box:', bounds)
    
    # Clip the pointfile with the polygon derived from the raster bounding box
    clipped = gpd.clip(pointfile, poly)
    
    # Save the clipped GeoDataFrame to a new shapefile, handling any errors
    output_path = f'/workspace/data/SGU/SFSI/SFSI/new_clips/{name}.shp'
    try:
        clipped.to_file(output_path)
        print(f'Successfully saved: {output_path}')
    except Exception as e:
        print(f'Error saving {output_path}: {e}')
        continue
