
import os
from osgeo import gdal
os.getcwd()
os.chdir('/national_datasets/SLUMarkfuktighetskarta')
os.getcwd()

data = gdal.Open("SLUMarkfuktighetskarta.tif")
gt = data.GetGeoTransform()
# get coordinates of upper left corner, which is the minimum x coordinate
xmin = gt[0]
# get coordinates of upper right corner, which is the maximnun y coordinate
ymax = gt[3]
#resolution of tile
res = gt[1]

print(gt)
print(xmin)
print(ymax)
print(res)
# determine total length of raster
xlen = res * data.RasterXSize
ylen = res * data.RasterYSize
print(xlen)
print(ylen)
# number of tiles in x and y direction
xdiv = 10
ydiv = 28

# size of a single tile
xsize = xlen/xdiv
ysize = ylen/ydiv

print(xsize)
print(ysize)
# create lists of x and y coordinates
xsteps = [xmin + xsize * i for i in range(xdiv+1)]
ysteps = [ymax - ysize * i for i in range(ydiv+1)]
# loop over min and max x and y coordinates
for i in range(xdiv):
    for j in range(ydiv):
        xmin = xsteps[i]
        xmax = xsteps[i+1]
        ymax = ysteps[j]
        ymin = ysteps[j+1]
        
        print("xmin: "+str(xmin))
        print("xmax: "+str(xmax))
        print("ymin: "+str(ymin))
        print("ymax: "+str(ymax))
        print("\n")
          # use gdal warp
        gdal.Warp("data/splitSLUMarkfuktighetskarta"+str(i)+str(j)+".tif", data, 
                  outputBounds = (xmin, ymin, xmax, ymax), dstNodata = -9999)
        
