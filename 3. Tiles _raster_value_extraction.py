import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd
#navigate to folder containing points cut to raster shape
allpoints = glob.glob('/workspace/data/SGU/SFSI/SFSI/new_clips/*.shp')

for point in allpoints:
    # opens the shapefile specified by the point variable and loads it as a GeoDataFrame. 
    pts = gpd.read_file(point)
    pts_name = point.split("/")[-1].split('.')[0]
    print('reading point tile number :', pts_name)
    # creates a list of coordinate pairs using the newx_point and newy_point columns of the pts GeoDataFrame. then uses zip() to pair up corresponding values from the two columns
    coords = [(x,y) for x, y in zip(pts.newx_point, pts.newy_point)]
    # opens a raster stack tile using the file name stored in the pts_name variable so it finds the correct tile
    src=rio.open('/workspace/data/SGU/orginal_fran_wl/CompositeBands/{:s}.tif'.format(pts_name))
    proper_col_name={1:"DEM",2:"EAS1ha", 3:"EAS10ha",4:"DI2m",
        5:"CVA",6: "SDFS",7: "DFME",8: "Rugged",9: "NMD",10:"SoilMap",11:"HKDepth",12:"SoilDepth", 13:"LandAge",
        14:"MSRM",17:"MED"}
    for i in range (1, src.count+1):
        if i in [15, 16]:#skip band 15, 16 which stand for soil, x,y 
            continue
        # extracts the values from the src raster at the coordinates for the current band i
        pts[proper_col_name[i]] = [x[i-1] for x in src.sample(coords)] # assigns the extracted values to a new column in the GeoDataFrame with the column name specified by proper_col_name[i]. x[i-1] ensures it only attaches the corresponding column value because bands are 0 indexed but column names are not 
        print(proper_col_name[i], ' attached!')
        #print(pts)
    pts.to_csv('/workspace/data/SGU/SFSI/SFSI/extracted_csv/output_{:s}.csv'.format(pts_name))#save back to a csv
    print('output {:s} , csv saved!'.format(pts_name))

    # Get a list of all CSV files in the folder
csv_files = glob.glob('/workspace/data/SGU/SFSI/SFSI/extracted_csv/*.csv')

# Initialize an empty DataFrame to store the concatenated data
concatenated_df = pd.DataFrame()

# Iterate over each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Concatenate the DataFrame with the existing data
    concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
    print('finished!')
# Save the concatenated data to a new CSV file
output_file = '/workspace/data/SGU/SFSI/SFSI/concatenated_data_original.csv'
concatenated_df.to_csv(output_file, index=False)
