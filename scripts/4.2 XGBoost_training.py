import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Import seaborn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
import csv
import time



# Define your main folder where you want to store the results
main_folder = '/workspace/data/SGU/SFSI/SFSI/XBG10x_akermark_7class/'

# Check if the main folder exists
if not os.path.exists(main_folder):
    # If it doesn't exist, create it and any missing parent folders
    os.makedirs(main_folder)
    print(f"Main folder '{main_folder}' was created.")
else:
    print(f"Main folder '{main_folder}' already exists.")

# Generate a timestamp to create a unique subfolder
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
sub_folder = os.path.join(main_folder, timestamp)
os.makedirs(sub_folder, exist_ok=True)  # Create the subfolder
print('Created subfolder.')

# Import training data
train = pd.read_csv('/workspace/data/Akermark/MASTER_train_plotxy.csv', sep= ',', decimal = '.')
test = pd.read_csv('/workspace/data/Akermark/MASTER_test_plotxy.csv', sep= ',', decimal = '.')
print('Read training and test data.')

le = LabelEncoder()
X_train = train.drop(columns=['REGION', 'GENERAL_TX','KARTTYP','QD_GENERAL_TX'], axis=1)
y_train = le.fit_transform(train['GENERAL_TX'])
X_test = test.drop(columns=['REGION', 'GENERAL_TX','KARTTYP','QD_GENERAL_TX'], axis=1)
y_test = le.fit_transform(test['GENERAL_TX'])
D_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
D_test =  xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
print(X_train)

# Create dataframes to store results
results_train = pd.DataFrame()
results_test = pd.DataFrame()
precision_recall_fscore_support_train_df = pd.DataFrame()
precision_recall_fscore_support_test_df = pd.DataFrame()

# Create a variable to store the best MCC score
best_mcc_score = -1  # Initialize with a value that guarantees replacement

# Create a variable to store the best model
best_model = None
# Initialize an empty list to store random states
random_states = []

target_names = ['coarse sed', 'coarse sed AGR', 'fine sed', 'fine sed AGR', 'peat', 'rock', 'till']
# Create dictionaries to store lists of the individual class precision, recall, f1 scores, and support for each class in each iteration
classwise_precision_per_iteration = {class_name: [] for class_name in target_names}
classwise_recall_per_iteration = {class_name: [] for class_name in target_names}
classwise_f1_per_iteration = {class_name: [] for class_name in target_names}
classwise_support_per_iteration = {class_name: [] for class_name in target_names}

# Define the number of iterations
num_iterations = 100

# Read random states from a text file
random_states_file = '/workspace/data/SGU/SFSI/SFSI/XBG10x_akermark_7class/20240308102531lidar/random_state.txt'
# Read random states from the text file so it is using the same random state as XGB
with open(random_states_file, 'r') as file:
   random_states = [int(line.strip()) for line in file]

# Loop for multiple iterations
training_times = []
for iteration, random_state in enumerate(random_states):
    print(f'Iteration {iteration + 1} of {num_iterations} using seed: {random_state}')

    # Declare parameters obtained from hyperparameter tuning using bayesian optimization
    params = {'subsample': 0.8021350553735852, 'lambda': 3.962846416422405, 'alpha': 4.421374345798305, 'booster': 'gbtree', 
            'max_depth': 11, 'eta': 0.28376636175248204, 'gamma': 1.2015387148071284, 'colsample_bytree': 0.7312808086252447, 'min_child_weight': 8,
              'num_class': 7, 'verbosity':1}
    params['seed'] = random_state  # Set the seed for randomness
    # Start the timer
    start_time = time.time()

     # Initialize and fit the model
    xgb_model = xgb.train(params, D_train )
    
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time for this iteration
    elapsed_time = end_time - start_time
    training_times.append(elapsed_time)

    y_train_pred = xgb_model.predict(D_train)
    y_test_pred = xgb_model.predict(D_test)
    
    print('Model fitting and prediction complete.')
    
    # Calculate evaluation metrics for training and testing
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    cohen_kappa_train = cohen_kappa_score(y_train, y_train_pred)
    cohen_kappa_test = cohen_kappa_score(y_test, y_test_pred)
    f1_train_w = f1_score(y_train, y_train_pred, average='weighted')
    f1_test_w = f1_score(y_test, y_test_pred, average='weighted')
    f1_train_uw = f1_score(y_train, y_train_pred, average=None)
    f1_test_uw = f1_score(y_test, y_test_pred, average=None)
    mcc_train = matthews_corrcoef(y_train, y_train_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)
    #precision_recall_fscore_support_train = precision_recall_fscore_support(y_train, y_train_pred, average = None)
    #precision_recall_fscore_support_test = precision_recall_fscore_support(y_test, y_test_pred, average = None)
    classification_report_test = classification_report(y_test, y_test_pred, target_names=target_names, digits=3, output_dict=True)

    precision_test, recall_test, f1_score_test, support_test = precision_recall_fscore_support(y_test, y_test_pred, average=None)


    # Assuming target_names is a list of class names
    for i, class_name in enumerate(target_names):
        classwise_precision_per_iteration[class_name].append(precision_test[i])
        classwise_recall_per_iteration[class_name].append(recall_test[i])
        classwise_f1_per_iteration[class_name].append(f1_score_test[i])
        classwise_support_per_iteration[class_name].append(support_test[i])

    # Create dataframes for this iteration's results
    iteration_results_train = pd.DataFrame({
        'Train Accuracy': [train_accuracy],
        'Cohen\'s Kappa (Train)': [cohen_kappa_train],
        'F1 Score (Train) weighted': [f1_train_w],
        'MCC (Train)': [mcc_train],
    })

    # Add separate columns for each class's unweighted F1 score
    for i, class_name in enumerate(target_names):
        iteration_results_train[f'F1_{class_name}'] = f1_train_uw[i]

    # Create dataframes for this iteration's results
    iteration_results_test = pd.DataFrame({
        'Test Accuracy': [test_accuracy],
        'Cohen\'s Kappa (Test)': [cohen_kappa_test],
        'F1 Score (Test) weighted': [f1_test_w],
        'MCC (Test)': [mcc_test],
    })

    # Add separate columns for each class's unweighted F1 score
    for i, class_name in enumerate(target_names):
        iteration_results_test[f'F1_{class_name}'] = f1_test_uw[i]

    # Concatenate the iteration results to the overall results dataframes
    results_train = pd.concat([results_train, iteration_results_train], ignore_index=True)
    results_test = pd.concat([results_test, iteration_results_test], ignore_index=True)


    print('Results dataframes updated.')

    # Check if the current model has a better MCC score
    if mcc_test > best_mcc_score:
        best_mcc_score = mcc_test
        best_model = xgb_model  # Update the best model with the current model

# Save the best model with pickle
model_file_path = os.path.join(sub_folder, 'best_model.json')
best_model.save_model(model_file_path)
print('best model saved.')

# Define a path for the seed file within the subfolder
state_path = os.path.join(sub_folder, 'random_state.txt')

# Save the random seeds to a text file in the subfolder
with open(state_path, 'w') as seed_file:
    seed_file.write('\n'.join(map(str, random_states)))
print('Random states saved.')

# Save the results to CSV
results_train_path = os.path.join(sub_folder, 'XGB_train_metrics.csv')
results_train.to_csv(results_train_path, index=False)
results_test_path = os.path.join(sub_folder, 'XGB_test_metrics.csv')
results_test.to_csv(results_test_path, index=False)
#precision_recall_fscore_support_test_path = os.path.join(sub_folder, 'XGB_precision_recall_fscore_support_test.csv')
#precision_recall_fscore_support_test.to_csv(precision_recall_fscore_support_test_path, index=False)

print('Results saved to CSV.')

# Create box plots to visualize the distribution of evaluation metrics for training and testing
plt.figure(figsize=(10, 5))

# Training metrics
plt.subplot(1, 2, 1)
plt.ylim(0.4, 1.)
plt.boxplot([results_train['Cohen\'s Kappa (Train)'], results_train['F1 Score (Train) weighted'], results_train['MCC (Train)']],
            labels=['Cohen\'s Kappa (Train)', 'F1 Score (Train) weighted', 'MCC (Train)'])
plt.title('Training Metrics')

# Testing metrics
plt.subplot(1, 2, 2)
plt.ylim(0.4, 1.)
plt.boxplot([results_test['Cohen\'s Kappa (Test)'], results_test['F1 Score (Test) weighted'], results_test['MCC (Test)']],
            labels=['Cohen\'s Kappa (Test)', 'F1 Score (Test) weighted', 'MCC (Test)'])
plt.title('Testing Metrics')

plt.tight_layout()

# Save the plots as images
box_plot_path = os.path.join(sub_folder, 'XGB10x_metrics_box_plots.png')
plt.savefig(box_plot_path)
print('Boxplot saved.')

## Create DataFrames from dictionaries
df_f1 = pd.DataFrame(classwise_f1_per_iteration)
df_precision = pd.DataFrame(classwise_precision_per_iteration)
df_recall = pd.DataFrame(classwise_recall_per_iteration)
df_support = pd.DataFrame(classwise_support_per_iteration)

# Specify file paths for saving CSV files
csv_f1_path = os.path.join(sub_folder,'classwise_f1_scores.csv')
csv_precision_path = os.path.join(sub_folder,'classwise_precision_scores.csv')
csv_recall_path = os.path.join(sub_folder,'classwise_recall_scores.csv')
csv_support_path = os.path.join(sub_folder,'classwise_support_values.csv')

# Save DataFrames to CSV files
df_f1.to_csv(csv_f1_path, index=False)
df_precision.to_csv(csv_precision_path, index=False)
df_recall.to_csv(csv_recall_path, index=False)
df_support.to_csv(csv_support_path, index=False)

print(f"Results saved to {csv_f1_path}, {csv_precision_path}, {csv_recall_path}, and {csv_support_path}.")

# Plot box plots for class-wise F1 scores
plt.figure(figsize=(10, 6))
df_f1.boxplot(rot=45, sym='k+', grid=False)
plt.title('Class-wise F1 Scores Box Plot')
plt.ylabel('F1 Score')
plt.xlabel('Class Name')
plt.tight_layout()
plt.show()

# Save the figure to a PNG file
box_plot_path = os.path.join(sub_folder, 'unweighted_f1_score_box_plot.png')
plt.savefig(box_plot_path)
print('Classwise f1 score box plot saved')

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f} seconds")
# Calculate the mean training time across all iterations
mean_training_time = sum(training_times) / num_iterations
print(f"Mean training time per iteration: {mean_training_time:.2f} seconds")

# Save the best model hyperparameters
best_model_params = best_model.get_params()

# Define the path for saving the hyperparameters and training time
hyperparams_training_time_path = os.path.join(sub_folder, 'best_model_hyperparams_and_training_time.txt')

# Write the hyperparameters and training time to the text file
with open(hyperparams_training_time_path, 'w') as f:
    f.write("Best Model Hyperparameters:\n")
    f.write(str(best_model_params))
    f.write("\n\nMean Training Time per Iteration: ")
    f.write(str(mean_training_time))
    f.write(" seconds\n")

print(f"Best model hyperparameters and training time saved to {hyperparams_training_time_path}.")

