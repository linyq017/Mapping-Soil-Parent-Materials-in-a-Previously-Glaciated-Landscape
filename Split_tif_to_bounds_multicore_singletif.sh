cd /workspace/data/SGU/orginal_fran_wl/rastertindex

# Define the number of parallel processes
num_processes=50

# Use xargs for parallel processing of shapefiles
ls *.shp | xargs -n 1 -P "$num_processes" bash -c $'
    shapefile=$1
    shapefile_name=$(basename "$shapefile" .shp)
    output_tif="/workspace/Lidardata01/Lin/Data/Indices/tiles/MINICURV20/${shapefile_name}.tif"
    tif="/workspace/Lidardata01/Lin/Data/Indices/resampled2m/MINICURV20.tif"
    
    # Check if the output TIFF file already exists
    if [ ! -f "$output_tif" ]; then
        gdalwarp -cutline "$shapefile" -crop_to_cutline "$tif" "$output_tif"
        echo "Cropped TIFF saved: $output_tif"
    else
        echo "Skipped processing for $shapefile_name, as $output_tif already exists."
    fi

    echo
' _
