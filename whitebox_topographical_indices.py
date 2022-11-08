import os
import argparse
try:
    import whitebox
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    wbt = WhiteboxTools()

parser = argparse.ArgumentParser(description='Extract topogrpahical incides from DEMs. ')

def main(input_path, output_path_hillshade, output_path_directional_relief, output_path_flow_accumulation):

    # setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]

    else:
        imgs = [input_path]

    
    for img_path in imgs:
        predicted = []
        print(img_path)
        img_name = os.path.basename(img_path).split('.')[0]
        
        hillshade =  os.path.join(output_path_hillshade,'{}.{}'.format(img_name + '_hillshade', 'tif'))
        relief = os.path.join(output_path_directional_relief,'{}.{}'.format(img_name + '_relief', 'tif'))
        flow_accumulation = os.path.join(output_path_flow_accumulation,'{}.{}'.format(img_name + '_flow', 'tif'))

        #This tool performs a hillshade operation (also called shaded relief) on an input digital elevation model (DEM) with multiple sources of illumination.
        wbt.multidirectional_hillshade(
            dem = img_path, 
            output = hillshade, 
            altitude=45.0, 
            zfactor=None, 
            full_mode=False
        )
        #Directional relief is an index of the degree to which a DEM grid cell is higher or lower than its surroundings. It is calculated by subtracting the elevation 
        # of a DEM grid cell from the average elevation of those cells which lie between it and the edge of the DEM in a specified compass direction.
        wbt.directional_relief(
            dem = img_path, 
            output = relief, 
            azimuth = 0.0,
            max_dist = None
        )
        #This tool is used to generate a flow accumulation grid (i.e. contributing area) using the D-infinity algorithm (Tarboton, 1997). This algorithm is an examples 
        # of a multiple-flow-direction (MFD) method because the flow entering each grid cell is routed to one or two downslope neighbour
        wbt.d_inf_flow_accumulation(
            i = img_path, 
            output = flow_accumulation, 
            out_type="Specific Contributing Area", 
            threshold=8
        )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to dem or folder of dems')
    parser.add_argument('output_path_hillshade', help = 'directory to store hillshade images')
    parser.add_argument('output_path_directional_relief', help = 'directory to store directional relief images')
    parser.add_argument('output_path_flow_accumulation', help = 'directory to store flow accumulation images')
    args = vars(parser.parse_args())
    main(**args)
