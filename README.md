# Mapping Soil Parent Materials in a Previously Glaciated Landscape: Potential for A Machine Learning Approach for Detailed Nationwide Mapping![image](https://github.com/user-attachments/assets/8366d00d-b5c3-414e-b507-91cffcdbf5d8)


This repository contains the code and data associated with our study titled "Mapping Soil Parent Materials in a Previously Glaciated Landscape: Potential for A Machine Learning Approach for Detailed Nationwide Mapping". The study explores the application of machine learning, specifically the Extreme Gradient Boosting (XGBoost) algorithm, to enhance surface deposit mapping in Sweden. Our model was trained on a dataset of 29,588 soil observations, with an additional hold-out set of 3,500 observations used for evaluation.

Key features used for training include terrain and hydrological indices derived from high-resolution airborne laser scanning data (2m resolution), supplemented with various ancillary map data. The XGBoost model trained on all available features achieved a Matthews Correlation Coefficient (MCC) of 0.56, outperforming the existing Quaternary Deposit Map. However, certain deposit types, such as highly variable sorted sediments, were more challenging to predict accurately compared to peat and till. To evaluate the model's robustness, it was validated across 28 physiographic regions in Sweden, revealing spatial variations in performance.

## Repository Structure

- `scripts/`: Python scripts for selected Lidar indices calculation, data processing, model training, and evaluation. 
- `dockerfile/`: Outputs of the models, including predictions and performance metrics.
- `README.md`: This file, providing an overview of the project.

## Installation and Setup

We recommend using a docker container to reproduce the results or run the code in this repository.

## Data Availability

The dataset used in this study includes 29,588 soil observations for training and 3,500 for evaluation. Due to sensitivity considerations of the Swedish Forest Soil Inventory plots, the raw data cannot be directly shared. Users are encouraged to use their own dataset.

## Usage
Usage
The Python scripts in the scripts/ directory are organized to perform various geospatial data processing tasks. Below is a brief overview of their functionality and how to execute them.

1. Resampling DEM to Different Resolutions
To resample Digital Elevation Models (DEMs) to different spatial resolutions, use the appropriate script in the scripts/ directory. This process adjusts the resolution of the DEM to meet specific analysis requirements.

2. Calculating Terrain Indices from LiDAR DEM
To compute terrain indices from LiDAR DEMs, you can use the example scripts provided. These scripts are numbered 2.1 through 2.5 and should be run in the terminal with the following syntax:
```bash
python script_name.py /path/to/input_directory /path/to/output_directory extra_argument
```
For example, to run script 2.3, run:
```bash
python /workspace/code/wbtools/relative_topographic_position.py /workspace/data/wbt/folder1/ /workspace/data/wbt/folder1_ruggedness/ 11
```

3. Extracting Raster Values to Points
Depending on your data format, you may need to extract raster values to point locations. This process varies based on the data types and formats you are working with:

- Single Raster: Extract values from a single raster file.
- Raster Tiles: Extract values from multiple raster tiles.
- Raster Stacks: Extract values from a stack of rasters.
- Point Shape File: Extract values based on point locations defined in a shapefile.
- Point Shape Files Clipped to Raster Bounds: Extract values for points within the bounds of the raster.

4. XGBoost Model Tuning, Training, and Map Prediction
The final step involves tuning, training, and making predictions with the XGBoost model. This process includes:

Model Tuning: Adjust hyperparameters to optimize model performance.
Training: Fit the model to your training dataset.
Map Prediction: Use the trained model to make predictions on new geospatial data.
SHAP value calculation: Calculate SHAP values for feature importance.

## Results and Discussion

In this study we trained two XGBoost models. The first model was trained only with LiDAR indices, the second model was trained with additional map data. The XGBoost model trained with LiDAR indices alongside map data achieved a Matthews Correlation Coefficient (MCC) of 0.56, indicating moderate predictive power. While the model successfully identified peat and till deposits, sorted sediments posed a greater challenge. Spatial validation revealed that model performance varied significantly across the 28 physiographic regions, highlighting the need for region-specific tuning or additional features.

For a detailed discussion of these results, refer to the [manuscript]((https://www.sciencedirect.com/science/article/pii/S2352009424001524?via%3Dihub)).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
