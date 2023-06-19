#first i created a txt containing only x and y of my points
awk -F ";" '{ if(NR>1) printf ("%i %i\n", $35, $34) }' /workspace/data/SGU/SFSI/SFSI/sfsi_correct_merged.csv > /workspace/data/SGU/SFSI/SFSI/x_y_merged.txt
#this creates a big tile index shape file with all tiles already in
cd /workspace/data/SGU/orginal_fran_wl/CompositeBands
gdaltindex  tindex.shp  *.tif
#rasterize tindex to find which tiles overlap with points
gdal_rasterize -ot Int32  -co COMPRESS=DEFLATE  -tr 100 100  -a "location"  /workspace/data/SGU/orginal_fran_wl/CompositeBands/tindex.shp     /workspace/data/SGU/orginal_fran_wl/CompositeBands/tindex.tif 
# the raster created will have the value of the tile number, when doing another gdallocationinfo, i can extract those numbers and save to a list
gdallocationinfo -geoloc -valonly /workspace/data/SGU/orginal_fran_wl/CompositeBands/tindex.tif  < /workspace/data/SGU/SFSI/SFSI/x_y_merged.txt > /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/tiflist_overlap.txt 
# add the ".tif" to the end of my tif list so it can be read
awk '{ print $1".tif" }' /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/tiflist_overlap.txt > /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/tiflist_overlaptif.txt

#create virtual raster with only the tiles that overlap with my points to reduce file size
cd /workspace/data/SGU/orginal_fran_wl/CompositeBands/
gdalbuildvrt -input_file_list /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/tiflist_overlaptif.txt output.vrt
# use gdallocationinfo to extract the bands i need and transpose the extraction into correct rows and columns
gdallocationinfo -geoloc -b 1  -b 2 -b 3 -b 4 -b 5 -b 6 -b 7 -b 8 -b 9 -b 10 -b 11 -b 12 -b 13 -b 14 -b 17  -valonly /workspace/data/SGU/orginal_fran_wl/CompositeBands/output.vrt < /workspace/data/SGU/SFSI/SFSI/x_y_merged.txt  |   awk -v BB=14 'ORS=NR%BB?FS:RS' >   /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/extract_allb_t.txt
# append headers to values extracted from composite band tiles
allb_header="DEM EAS1ha EAS10ha DI2m CVA SDFS DFME Rugged SoilMap HKDepth SoilDepth LandAge MSRM MED"
echo $allb_header > /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/extract_allb_header.txt
cat  /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/extract_allb_T.txt     >> /workspace/data/SGU/orginal_fran_wl/CompositeBands_txt/extract_allb_header.txt
