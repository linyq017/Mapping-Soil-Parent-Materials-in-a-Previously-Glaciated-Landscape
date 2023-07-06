import os
import rasterio

def extract_band(input_folder, output_folder, band_to_extract):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the input raster
            with rasterio.open(input_path) as src:
                # Read the specified band
                band_data = src.read(band_to_extract)  # Bands are 1-based index in rasterio

                # Create the output raster with the same metadata as the input raster
                profile = src.profile
                profile.update(count=1)  # Set the number of bands to 1

                # Write the extracted band to the output raster
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(band_data, 1)

# Usage example:
input_folder = "/workspace/data/SGU/enkoping/overlap"
output_folder = "/workspace/data/SGU/enkoping/overlap_dem"
band_to_extract = 1  
extract_band(input_folder, output_folder, band_to_extract)
