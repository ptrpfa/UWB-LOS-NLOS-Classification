from config import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss, mean_squared_error
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
def classifier_metrics(list_y, list_pred):
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
    return results