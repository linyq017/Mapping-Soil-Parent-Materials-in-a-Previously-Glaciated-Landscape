
cd /workspace/data/SGU/orginal_fran_wl/rastertindex

# Define the number of parallel processes
num_processes=100

# Use ls and xargs for parallel processing of shapefiles
ls *.shp | xargs -n 1 -P "$num_processes" bash -c $'  # -n is number of arguments
    shapefile=$1
    shapefile_name=$(basename "$shapefile" .shp)
    output_tif="/workspace/Lidardata01/Lin/Data/Indices/tiles/CVA50/${shapefile_name}.tif"
    tif="/workspace/Lidardata01/Lin/Data/Indices/resampled2m/CVA50.tif"
    
    gdalwarp -cutline "$shapefile" -crop_to_cutline "$tif" "$output_tif"
    
    echo "Cropped TIFF saved: $output_tif"
    echo
' _
