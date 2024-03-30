from config import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss, mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

# Function to serialise an object into a pickle file
def save_to_pickle(file_name, save_data, complete_path=True):
    file_name_with_extension = file_name + ".pkl"
    complete_file_path = f'{EXPORT_FOLDER}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'wb') as file:
        pickle.dump(save_data, file)

# Function to deserialise a pickle file
def load_from_pickle(file_name, complete_path=True):
    file_name_with_extension = file_name + ".pkl"
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
def plot_histogram(df, features):
    # Plot numerical features with respect to the target variable
    plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features, start=1):
        plt.subplot(len(features)//2 + 1, 2, i)
        sns.histplot(data=df, x=feature, hue='NLOS', kde=True, stat='density', common_norm=False)
        plt.title(f'Distribution of {feature} by NLOS')
        plt.xlabel(feature)
        plt.ylabel('Density')
        
        # Calculate skewness for LOS and NLOS classes
        los_skewness = df[df['NLOS'] == 0][feature].skew()
        nlos_skewness = df[df['NLOS'] == 1][feature].skew()

        # Annotate skewness values on the plot
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