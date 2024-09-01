import os
import glob
import whitebox
import argparse

# Initialize WhiteboxTools
try:
    import whitebox
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = whitebox.WhiteboxTools()
except ImportError:
    from WBT.whitebox_tools import WhiteboxTools
    whitebox.download_wbt(linux_musl=True, reset=True)
    wbt = WhiteboxTools()

def resample(input_path, output_path, cell_size):
    """
    Resample all .tif files in the input directory using WhiteboxTools.

    Parameters:
    input_path (str): Path to the directory containing input .tif files.
    output_path (str): Path to the directory where processed files will be saved.
    cell_size (float): Desired cell size for the output raster files.
    """
    # Ensure the output directory exists
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory '{output_path}' is ready.")
    except Exception as e:
        print(f"Error creating output directory '{output_path}': {e}")
        return

    # Find all .tif files in the input directory
    tif_files = glob.glob(os.path.join(input_path, '*.tif'))
    
    if not tif_files:
        print(f"No .tif files found in input directory '{input_path}'.")
        return

    for tif_file in tif_files:
        # Define the output file path
        output_file = os.path.join(output_path, os.path.basename(tif_file))
        
        print(f"Processing {tif_file}...")
        try:
            # Apply the resample tool
            wbt.resample(
                inputs=tif_file, 
                output=output_file, 
                cell_size=float(cell_size), 
                method="cc"  # Cubic convolution resampling method
            )
            print(f"Completed processing {tif_file}")
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")

    print("All files processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resample all .tif files in the specified directory using the WhiteboxTools resample tool.',
        formatter_class=argparse.HelpFormatter
    )
    parser.add_argument('input_path', help='Path to the directory containing input .tif files.')
    parser.add_argument('output_path', help='Path to the directory where processed files will be saved.')
    parser.add_argument('cell_size', type=float, help='Desired cell size for the output raster files.')
    args = parser.parse_args()
    
    resample(args.input_path, args.output_path, args.cell_size)
