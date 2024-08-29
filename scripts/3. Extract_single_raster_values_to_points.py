import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd

def extract_raster_values(pointfile_path, tif_folder):
    """
    Extract raster values from TIFF files at point locations specified in a shapefile.

    Parameters:
    - pointfile_path (str): Path to the shapefile containing point locations.
    - tif_folder (str): Path to the folder containing TIFF files.

    Returns:
    - GeoDataFrame with extracted raster values.
    """
    # Load the point vector file
    try:
        pointfile = gpd.read_file(pointfile_path)
    except Exception as e:
        raise RuntimeError(f"Error reading shapefile: {e}")

    coords = list(zip(pointfile.newx_point, pointfile.newy_point))

    # Ensure TIFF files exist in the specified folder
    if not os.path.exists(tif_folder):
        raise FileNotFoundError(f"TIFF folder not found: {tif_folder}")

    # Loop through TIFF files and extract raster values
    for tif_file in os.listdir(tif_folder):
        if tif_file.endswith('.tif'):
            tif_path = os.path.join(tif_folder, tif_file)
            print(f"Reading file: {tif_file}")

            try:
                with rio.open(tif_path) as src:
                    # Extract raster values at the point locations
                    values = [src.sample([coord])[0] for coord in coords]
                    values = [value[0] if value.size > 0 else np.nan for value in values]
                    # Add raster values to the GeoDataFrame
                    pointfile[tif_file] = values
            except Exception as e:
                print(f"Error processing file {tif_file}: {e}")

    print("Extraction completed.")
    return pointfile

if __name__ == '__main__':
    pointfile_path = '/workspace/data/SGU/SFSI/project_shapefile/sfsi3.shp'
    tif_folder = '/workspace/data/wbt/newtifs'

    # Extract raster values and get updated GeoDataFrame
    updated_gdf = extract_raster_values(pointfile_path, tif_folder)

    # Optionally save the updated GeoDataFrame to a new file
    output_path = '/workspace/data/SGU/SFSI/project_shapefile/updated_sfsi3.shp'
    updated_gdf.to_file(output_path)
    print(f"Updated GeoDataFrame saved to {output_path}")
