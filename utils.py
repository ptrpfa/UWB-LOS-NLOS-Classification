from config import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import skew, gaussian_kde, linregress
from scipy.linalg import LinAlgError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, itertools

# Prepare dictionary containing each feature variant's exported file
dict_noncir = {
    'noncir': 'noncir.pkl',
    'noncir_mm_scaled': 'noncir_mm_scaled.pkl',
    'noncir_ss_scaled': 'noncir_ss_scaled.pkl',
    'noncir_trimmed': 'trimmed_noncir.pkl',
    'noncir_mm_scaled_trimmed': 'trimmed_noncir_mm_scaled.pkl',
    'noncir_ss_scaled_trimmed': 'trimmed_noncir_ss_scaled.pkl',
}

dict_cir = {
    'cir': 'cir.pkl',
    'cir_mm_scaled': 'cir_mm_scaled.pkl',
    'cir_ss_scaled': 'cir_ss_scaled.pkl',
    'cir_stats': 'cir_stats.pkl',
    'cir_stats_mm_scaled': 'cir_stats_mm_scaled.pkl',
    'cir_stats_ss_scaled': 'cir_stats_ss_scaled.pkl',
    'cir_stats_trimmed': 'trimmed_cir_stats.pkl',
    'cir_stats_mm_scaled_trimmed': 'trimmed_cir_stats_mm_scaled.pkl',
    'cir_stats_ss_scaled_trimmed': 'trimmed_cir_stats_ss_scaled.pkl',
    'cir_pca': 'cir_pca.pkl',
    'cir_pca_mm_scaled': 'cir_pca_mm_scaled.pkl',
    'cir_pca_ss_scaled': 'cir_pca_ss_scaled.pkl',
}

# Function to serialise an object into a pickle file
def save_to_pickle(file_name, save_data, complete_path=True):
    file_name_with_extension = file_name + ".pkl"
    complete_file_path = f'{EXPORT_FOLDER}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'wb') as file:
        pickle.dump(save_data, file)

# Function to deserialise a pickle file
def load_from_pickle(file_name, complete_path=True):
    if(".pkl" not in file_name):
        file_name_with_extension = file_name + ".pkl"
    else:
        file_name_with_extension = file_name
    complete_file_path = f'{EXPORT_FOLDER}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

# Function to return the performance for a classifier
def classifier_metrics(list_y, list_pred, print_results=False):
    # Obtain metrics
    results = {
        "accuracy": accuracy_score(list_y, list_pred),
        "precision": precision_score(list_y, list_pred, average='macro'),
        "recall": recall_score(list_y, list_pred, average='macro'),
        "f1": f1_score(list_y, list_pred, average='macro'),
        "mcc": matthews_corrcoef(list_y, list_pred),
        "mse": mean_squared_error(list_y, list_pred),
        "kappa": cohen_kappa_score(list_y, list_pred),
        "hamming_loss_val": hamming_loss(list_y, list_pred),
        "cm": confusion_matrix(list_y, list_pred),
        "class_report": classification_report(list_y, list_pred),
    }
    
    if(print_results):
        print("Accuracy:", results['accuracy'])                                    # Model Accuracy: How often is the classifier correct
        print("Precision:", results['precision'])                                  # Model Precision: what percentage of positive tuples are labeled as such?
        print("Recall:", results['recall'])                                        # Model Recall: what percentage of positive tuples are labelled as such?
        print("F1 Score:", results['f1'])                                          # F1 Score: The weighted average of Precision and Recall
        print("Mean Squared Error (MSE):", results['mse'])                         # Mean Squared Error (MSE): The average of the squares of the errors
        print("Matthews Correlation Coefficient (MCC):", results['mcc'])           # Matthews Correlation Coefficient (MCC): Measures the quality of binary classifications
        print("Cohen's Kappa:", results['kappa'])                                  # Cohen's Kappa: Measures inter-rater agreement for categorical items    
        print("Hamming Loss:", results['hamming_loss_val'], end='\n\n')            # Hamming Loss: The fraction of labels that are incorrectly predicted
        print("Confusion Matrix:\n", results['cm'], end="\n\n")
        print("Classification Report:\n", results['class_report'], end="\n\n\n")
        
    return results

# Function to plot feature-NLOS histogram
def plot_histogram(df):
    # Get non-class features
    features = [col for col in df.columns if col != 'NLOS']
    plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features, start=1):
        plt.subplot(len(features)//2 + 1, 2, i)
        try:
            sns.histplot(data=df, x=feature, hue='NLOS', kde=True, stat='density', common_norm=False)
        except LinAlgError as e:
            # print(f"Warning: {e}")
            sns.histplot(data=df, x=feature, hue='NLOS', stat='density', common_norm=False)
        plt.title(f'Distribution of {feature} by NLOS')
        plt.xlabel(feature)
        plt.ylabel('Density')
        los_skewness = df[df['NLOS'] == 0][feature].skew()
        nlos_skewness = df[df['NLOS'] == 1][feature].skew()
        plt.text(0.9, 0.9, f'Skewness (LOS): {los_skewness:.2f}\nSkewness (NLOS): {nlos_skewness:.2f}', 
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

# Function to plot feature-NLOS boxplot with different colors for each class and a custom legend
def plot_box_plot(df, features):
    # Define colors for each class
    palette = {0: "skyblue", 1: "orange"}
    
    # Plot numerical features with respect to the target variable
    plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features, start=1):
        plt.subplot(len(features)//2 + 1, 2, i)
        sns.boxplot(x='NLOS', y=feature, hue='NLOS', data=df, palette=palette, dodge=False, legend=False)
        plt.title(f'Boxplot of {feature}')
        plt.xlabel('Class')
        plt.ylabel(feature)
    
    # Create a legend for the plot
    handles = [plt.Rectangle((0,0),1,1, color=palette[label]) for label in palette.keys()]
    labels = ["LOS", "NLOS"]
    plt.figlegend(handles, labels, loc='lower right', title="Class")
    
    plt.tight_layout()
    plt.show()
    
# Function to plot feature-feature relationship scatter plots, with regards to LOS/NLOS
def plot_feature_relationship(df):
    # Filters for each class
    nlos_class = (df['NLOS'] == 1)
    los_class = (df['NLOS'] == 0)

    # Create dictionary of feature pair combinations, excluding NLOS
    dict_combinations = {}
    for combination in itertools.combinations(df.columns, 2):
        column1, column2 = combination
        if column1 not in dict_combinations:
            dict_combinations[column1] = []
        dict_combinations[column1].append((column1, column2))
    del dict_combinations['NLOS']

    # Plot each group of combinations separately
    for column, combinations in dict_combinations.items():
        fig, axes = plt.subplots(1, len(combinations), figsize=(20, 4))  # Adjust figsize as needed
        fig.suptitle(f"Plots for {column}", fontsize=16)
        if len(combinations) == 1:
            axes = [axes]  # Ensure axes is a list to handle single subplot case
        for i, combination in enumerate(combinations):
            ax = axes[i]  # Access the axes
            ax.scatter(df[nlos_class][combination[0]], df[nlos_class][combination[1]], c="yellow", s=20, edgecolor='k')
            ax.scatter(df[los_class][combination[0]], df[los_class][combination[1]], c="blue", s=20, edgecolor='k')
            ax.set_xlabel(combination[0])
            ax.set_ylabel(combination[1])
            ax.set_title(f"{combination[0]} vs {combination[1]}")
        plt.tight_layout()
        plt.show()

# Function to plot feature-feature correlation matrix
def plot_correlation_matrix(df):
    corr_df = df.drop(columns='NLOS')
    correlation_matrix = corr_df.corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Selected Features')
    plt.show()

    # Find pairs of features with high correlation coefficients
    high_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    if high_corr_pairs:
        print("Pairs of features with high correlation coefficients:")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.2f}")
    else:
        print("No pairs of features with high correlation coefficients found.")
        
def get_top_features(df):
    X = df.drop(columns='NLOS')
    y = df['NLOS']

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    # Fit the model
    rf_classifier.fit(X, y)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame based on feature importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the top 20 most important features
    return feature_importance_df