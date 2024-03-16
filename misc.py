import pickle 

""" Miscellaneous Functions """
# Function to serialise an object into a pickle file
file_path = "./dataset/Processed_Dataset/"

def save_to_pickle(file_name, save_data):
    file_name_with_extension = file_name + ".pkl"
    complete_file_path = file_path + file_name_with_extension

    with open(complete_file_path, 'wb') as file:
        pickle.dump(save_data, file)

# Function to deserialise a pickle file
def load_from_pickle(file_name):
    file_name_with_extension = file_name + ".pkl"
    complete_file_path = file_path + file_name_with_extension
    
    with open(complete_file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data