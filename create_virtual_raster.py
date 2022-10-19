from osgeo import gdal, osr
import glob
import os

#composites = [file for file in os.listdir('data/SGU/orginal_fran_wl/tif/') if file.endswith ('.tif')]
compolist = glob.glob('data/SGU/orginal_fran_wl/CompositeBands/*.tif')
#print(compolist)
#print(composites)
vrt = gdal.BuildVRT('data/SGU/orginal_fran_wl/tif/merged2.vrt', compolist)
vrt.FlushCache()#otherwise .vrt sometimes wont show up in the folder
vrt.ReasAsArray()#can be done if array size is not too big

print('virtual raster created!')
