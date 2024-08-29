# Evaluating Machine Learning Methods for Predicting Surface Deposits Across Physiographic Regions in Sweden

This repository contains the code and data associated with our study titled "Evaluating Machine Learning Methods for Predicting Surface Deposits Across Physiographic Regions in Sweden." The study explores the application of machine learning, specifically the Extreme Gradient Boosting (XGBoost) algorithm, to enhance surface deposit mapping in Sweden. Our model was trained on a dataset of 29,588 soil observations, with an additional hold-out set of 3,500 observations used for evaluation.

Key features used for training include terrain and hydrological indices derived from high-resolution airborne laser scanning data (2m resolution), supplemented with various ancillary map data. The XGBoost model trained on all available features achieved a Matthews Correlation Coefficient (MCC) of 0.56, outperforming the existing Quaternary Deposit Map. However, certain deposit types, such as highly variable sorted sediments, were more challenging to predict accurately compared to peat and till. To evaluate the model's robustness, it was validated across 28 physiographic regions in Sweden, revealing spatial variations in performance.

## Repository Structure

- `data/`: Contains a subset of anonymized data used in the study.
- `scripts/`: Python scripts for selected Lidar indices calculation, data processing, model training, and evaluation.
- `dockerfile/`: Outputs of the models, including predictions and performance metrics.
- `figures/`: Visualizations and plots generated during the study.
- `README.md`: This file, providing an overview of the project.

## Installation and Setup

To reproduce the results or run the code in this repository, we recommend using a docker container.

## Data Availability

The dataset used in this study includes 29,588 soil observations for training and 3,500 for evaluation. Due to sensitivity considerations, the raw data cannot be directly shared. However, a subset of anonymized data is available for testing purposes. 

## Usage

To calculate terrain indices from LiDAR dem, run the following scripts in the terminal:
```bash
python script_name.py /path/to/input_directory /path/to/output_directory


To train the XGBoost model using the provided dataset, run the following scripts:

## Results and Discussion

In this study we trained two XGBoost models. The first model was trained only with LiDAR indices, the second model was trained with additional map data. The XGBoost model trained with LiDAR indices alongside map data achieved a Matthews Correlation Coefficient (MCC) of 0.56, indicating moderate predictive power. While the model successfully identified peat and till deposits, sorted sediments posed a greater challenge. Spatial validation revealed that model performance varied significantly across the 28 physiographic regions, highlighting the need for region-specific tuning or additional features.

For a detailed discussion of these results, refer to the [manuscript](link-to-manuscript).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
