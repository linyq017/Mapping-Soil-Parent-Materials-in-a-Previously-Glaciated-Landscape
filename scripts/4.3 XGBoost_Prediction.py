import xgboost as xgb
from osgeo import gdal, gdal_array
import os
import numpy as np
import pandas as pd
import rasterio as rio

# Define folder paths
raster_composite_folder = "/workspace/data/composite"
indices_folder = '/workspace/data/NewIndices'
output_folder = '/workspace/data/XGB_output'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained XGBoost model from a JSON file
xgb_model = xgb.Booster(model_file='/workspace/data/model/best_model.json')

# Iterate over the raster files in the composite folder
for raster_file_name in os.listdir(raster_composite_folder):
    raster_file_path = os.path.join(raster_composite_folder, raster_file_name)

    # Process only TIFF files
    if raster_file_name.endswith('.tif'):
        # List to hold raster bands
        bands_list = []

        # Open the raster file using GDAL
        ds = gdal.Open(raster_file_path)

        # Read all bands from the raster file
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            print(f'Appending: {raster_file_name}, band {i}')
            bands_list.append(band.ReadAsArray())
        
        # Process additional raster bands from the indices folder
        for subfolder in os.listdir(indices_folder):
            subfolder_path = os.path.join(indices_folder, subfolder)
            if os.path.isdir(subfolder_path):
                subfolder_file_path = os.path.join(subfolder_path, raster_file_name)
                if os.path.exists(subfolder_file_path):
                    band = gdal_array.LoadFile(subfolder_file_path)
                    print(f'Appending: {raster_file_name} from subfolder {subfolder}')
                    bands_list.append(band)

        # Convert the list of bands to a numpy array
        all_data = np.array(bands_list)
        print(f"Array shape before reshape: {all_data.shape}")
        print(f"Total number of elements in the array: {all_data.size}")

        # Reshape the array to match the model's input dimensions
        all_data = all_data.reshape(-1, all_data.shape[-1])  # Reshape to match the number of columns
        print(f"Array shape after reshape: {all_data.shape}")

        # Create a DataFrame from the numpy array
        columns = ['DEM', 'EAS1ha', 'EAS10ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 'Rugged', 'NMD',
                   'SoilMap', 'HKDepth', 'SoilDepth', 'LandAge', 'MSRM', 'x', 'y', 'MED',
                   'ANVAD20_15', 'CVA20', 'CVA50', 'Directiona', 'DownslopeI',
                   'Geomorphon', 'MAXCURV20', 'MAXCURV50', 'MED20', 'MED50',
                   'MINICURV20', 'MaxDownslo', 'NDVI', 'ProfileCur',
                   'RELTOPOPOS', 'SLOPE20', 'SLOPE50', 'TWI20']
        df_data = pd.DataFrame(all_data, columns=columns)

        # Ensure columns are in the same order as required by the model
        required_columns = ['x', 'y', 'DEM', 'EAS1ha', 'EAS10ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 
                            'Rugged', 'NMD', 'SoilMap', 'HKDepth', 'SoilDepth', 'LandAge', 
                            'MSRM', 'MED', 'CVA20', 'CVA50', 'MAXCURV20', 'MAXCURV50', 
                            'MINICURV20', 'SLOPE20', 'SLOPE50', 'MED20', 'MED50', 
                            'ANVAD20_15', 'Directiona', 'DownslopeI', 'Geomorphon', 
                            'MaxDownslo', 'NDVI', 'ProfileCur', 'RELTOPOPOS', 'TWI20']
        df_data = df_data[required_columns]

        # Create a DMatrix for prediction
        dmatrix = xgb.DMatrix(df_data) 
        predictions = xgb_model.predict(dmatrix)

        # Reshape predictions to the original raster dimensions
        predictions = predictions.reshape(1250, 1250)
        
        # Open an original raster to copy its metadata
        with rio.open(raster_file_path) as src:
            metadata = src.profile
            metadata.update(count=1)  # Update metadata to reflect single-band output

        # Save the predictions as a new raster file with the same metadata
        output_file = os.path.join(output_folder, raster_file_name)
        with rio.open(output_file, 'w', **metadata) as dst:
            dst.write(predictions, 1)
            print(f'Raster saved: {output_file}')
