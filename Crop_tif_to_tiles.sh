%%bash

# Set the paths to the input TIFF folder and shapefile folder
shapefile_folder="/workspace/data/SGU/orginal_fran_wl/rastertindex"
tif_folder="/workspace/data/wbt/newtifs"


# Iterate over the TIFF files in the input folder
for tif_file in "$tif_folder"/*.tif; do
    echo "Processing TIFF: $tif_file"

    # Get the base name of the TIFF file (without the extension)
    tif_name=$(basename "$tif_file" .tif)

    # Create the output folder based on the TIFF name
    output_folder="/workspace/data/wbt/newtifs/$tif_name"
    mkdir -p "$output_folder"

    # Iterate over the shapefiles in the shapefile folder
    for shapefile in "$shapefile_folder"/*.shp; do
        echo "Processing shapefile: $shapefile"

        # Get the base name of the shapefile (without the extension)
        shapefile_name=$(basename "$shapefile" .shp)

        # Set the output TIFF path
        output_tif="$output_folder/${shapefile_name}.tif"

        # Use gdalwarp to crop the TIFF using the shapefile geometry
        gdalwarp -cutline "$shapefile" -crop_to_cutline "$tif_file" "$output_tif"

        echo "Cropped TIFF saved: $output_tif"
        echo
    done
done
