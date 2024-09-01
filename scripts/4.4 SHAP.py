import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the XGBoost model from a JSON file
xgb_model = xgb.Booster(model_file='/workspace/data/SGU/SFSI/SFSI/XBG10x_akermark_7class/20240326093033LIDAR_noNDVI/best_model.json')

# Initialize LabelEncoder to encode target variable
le = LabelEncoder()

# Set Seaborn style and context for plots
sns.set_style("ticks")
sns.set_context("paper")

# Load test data from CSV file
test = pd.read_csv('/workspace/data/Akermark/MASTER_test_plotxy.csv', sep=',', decimal='.')
print("Columns in test data:", test.columns)

# Prepare test features (X) and target variable (y)
X_test = test.drop(columns=['x', 'y', 'NMD', 'SoilMap', 'HKDepth', 'SoilDepth', 'LandAge', 'NDVI', 'REGION', 'KARTTYP', 'GENERAL_TX', 'QD_GENERAL_TX'])
y_test = le.fit_transform(test['GENERAL_TX'])

# Create a SHAP explainer object with the XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test data
shap_values = explainer.shap_values(X_test)

# Define class names and colors for visualization
classes = ['Coarse sed FOR', 'Coarse sed AGR', 'Fine sed FOR', 'Fine sed AGR', 'Peat', 'Rock outcrops', 'Till']
colors = ["#6F8C57", "#446589", '#EFD460', '#B0633F', '#926CCF', '#EA5A94', "#9C9C9C"]

# Determine the order of classes based on SHAP value magnitude
class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])

# Create a colormap from the class colors
cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

# Define feature names for better readability in plots
feature_display_names = [
    'Digital Elevation Model', 'Elevation above Stream (1ha network)', 'Elevation above Stream (10ha network)', 
    'Downslope Index 2m drop', 'Circular Variance of Aspect', 'Standard Deviation of Slope', 'Deviation from Mean Elevation',
    'Terrain Ruggedness Index', 'Multiscale Roughness Magnitude', 'Maximum Elevation Deviation', 
    'Circular Variance of Aspect 20m', 'Circular Variance of Aspect 50m', 'Maximum Curvature 20m', 'Maximum Curvature 50m', 
    'Minimum Curvature 20m', 'Slope 20m', 'Slope 50m', 'Maximum Elevation Deviation 20m', 'Maximum Elevation Deviation 50m', 
    'Average Normal Vector Angular Deviation 20m', 'Directional Relief', 'Downslope Index with 2m drop 20m', 
    'Geomorphons 20m', 'Max Downslope Elevation Change 20m', 'Normalized Difference Vegetation Index', 
    'Profile Curvature 20m', 'Relative topographic positions 20m', 'Topographic Wetness Index 20m'
]

# Set the maximum number of features to display
max_display_features = 15

# Create a figure with the desired size
plt.figure(figsize=(12, 6))  # Adjust width and height as needed

# Generate the SHAP summary plot
shap.summary_plot(
    shap_values, 
    X_test, 
    feature_names=feature_display_names, 
    color=cmap, 
    class_names=classes, 
    show=False, 
    max_display=max_display_features, 
    plot_size=(9, 5)
)

# Customize plot appearance
plt.xticks(fontsize=10)  # Adjust font size for x-axis ticks
plt.yticks(fontsize=10)  # Adjust font size for y-axis ticks

# Access and customize the legend font size
legend = plt.gca().get_legend()
if legend:
    for label in legend.get_texts():
        label.set_fontsize(10)  # Set font size for legend labels

plt.xlabel('Mean SHAP Value (average impact on XGBoost output)', fontsize=9.5)

# Save the plot with high resolution
plt.savefig('/workspace/data/manuscript1plots/shap_summary_plot_lidaronly.png', dpi=600)

# Display the plot
plt.show()
