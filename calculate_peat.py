import os
import numpy as np
import rasterio as rio
from matplotlib import pyplot

#path = '/Volumes/Extension_100TB/Lin/data/testdata\\'
path = 'data/split/'
#output_path = '/Volumes/Extension_100TB/Lin/data/droplake\\'
output_path = 'data/49_tiles/'

for file in os.listdir(path):
    if file.endswith('.TIF'):
        print(file)
        filepath = path+file
        output_file = output_path+'8bit_'+file
        
        ####read raster into numpy
        with rio.open(filepath) as src:
            ##read band 1 from raster AS FLOAT! original datatype 8 bit integer which doesnt calculate correctly
            ras_array = src.read(1).astype('float32')
            print('original array:',ras_array)
            #if original array contains nodata, change to 0, otherwise keep original value
            power2 = np.power(ras_array, 2)
            power3 = np.power(ras_array, 3)
            new_vals=(6.414594984 + (0.667350616*ras_array) + ((-0.021458236)* power2) + (0.000229229 * power3))
            file_s = np.where((ras_array < 101), new_vals, 0)
            print('maximum value after calculation:',np.max(file_s))
            print('new array:',file_s)
            #round up numbers
            ras_array_new = np.rint(file_s)
            print('maximum value in new rounded-up array:',np.max(ras_array_new),'minimum value in new rounded-up array:',np.min(ras_array_new))

            #change data type to 8 bit
            ras_meta = src.profile
            ras_meta.update(nodata=0,dtype =rio.uint8, compress = 'lzw')
            print('new metadata:',ras_meta)

####save numpy back to raster
        with rio.open(output_file, 'w', **ras_meta) as dst:
            dst.write(ras_array_new, 1)
            print('tile {:s} , raster saved!'.format(file))
