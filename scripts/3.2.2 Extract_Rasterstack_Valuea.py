import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd

"""
This script extracts raster values at specific point locations from multiple raster tiles and 
saves the results into individual CSV files. Finally, it concatenates all these CSV files into 
a single CSV file.

Usage:
- This script is useful for extracting raster data for specific point locations (e.g., for 
  environmental or geological studies) and then combining these extracted values into one comprehensive dataset.
"""

# Navigate to the folder containing the point shapefiles that have been clipped to raster extents
allpoints = glob.glob('/workspace/data/SGU/SFSI/SFSI/new_clips/*.shp')

# Iterate over each shapefile in the directory
for point in allpoints:
    # Load the shapefile as a GeoDataFrame
    pts = gpd.read_file(point)
    
    # Extract the name of the shapefile (without extension) to match it with the raster tile
    pts_name = os.path.splitext(os.path.basename(point))[0]
    print('Reading point tile number:', pts_name)
    
    # Create a list of coordinate pairs from the 'newx_point' and 'newy_point' columns
    coords = [(x, y) for x, y in zip(pts.newx_point, pts.newy_point)]
    
    # Open the corresponding raster file using the base name of the shapefile
    src = rio.open(f'/workspace/data/SGU/orginal_fran_wl/CompositeBands/{pts_name}.tif')
    
    # Dictionary mapping band numbers to descriptive column names
    proper_col_name = {
        1: "DEM", 2: "EAS1ha", 3: "EAS10ha", 4: "DI2m",
        5: "CVA", 6: "SDFS", 7: "DFME", 8: "Rugged", 9: "NMD", 10: "SoilMap",
        11: "HKDepth", 12: "SoilDepth", 13: "LandAge", 14: "MSRM", 17: "MED"
    }
    
    # Iterate over each band in the raster file
    for i in range(1, src.count + 1):
        # Skip bands 15 and 16 (assuming these are not needed)
        if i in [15, 16]:
            continue
        
        # Extract raster values at the specified coordinates and assign them to the GeoDataFrame
        pts[proper_col_name[i]] = [x[i - 1] for x in src.sample(coords)]
        print(f'{proper_col_name[i]} attached!')
    
    # Save the GeoDataFrame with the extracted values to a CSV file
    output_csv = f'/workspace/data/SGU/SFSI/SFSI/extracted_csv/output_{pts_name}.csv'
    pts.to_csv(output_csv, index=False)
    print(f'Output {pts_name} CSV saved!')

# After processing all shapefiles, concatenate all resulting CSV files into a single CSV

# Get a list of all CSV files in the folder
csv_files = glob.glob('/workspace/data/SGU/SFSI/SFSI/extracted_csv/*.csv')

# Initialize an empty DataFrame to store the concatenated data
concatenated_df = pd.DataFrame()

# Iterate over each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Concatenate the DataFrame with the existing data
    concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

# Save the concatenated data to a new CSV file
output_file = '/workspace/data/SGU/SFSI/SFSI/concatenated_data_original.csv'
concatenated_df.to_csv(output_file, index=False)
print('All files concatenated and saved!')
