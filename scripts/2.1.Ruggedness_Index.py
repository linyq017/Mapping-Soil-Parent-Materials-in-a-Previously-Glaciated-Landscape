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

def ruggedness_index(input_path, output_path):
    """Apply ruggedness index tool to each .tif file in the input directory."""
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
            # Apply the ruggedness_index tool
            wbt.ruggedness_index(
                dem=tif_file,
                output=output_file,
            )
            print(f"Completed processing {tif_file}")
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")

    print("All files processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply maximal curvature tool to DEM tiles.',
        formatter_class=argparse.HelpFormatter
    )
    parser.add_argument('input_path', help='Path to the directory containing input .tif files.')
    parser.add_argument('output_path', help='Path to the directory where processed files will be saved.')
    
    args = parser.parse_args()
    
    ruggedness_index(args.input_path, args.output_path)
