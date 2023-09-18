import pickle


pickle_file_path = "C:/Users/nikod/Documents/RProjects/hivProject/pythonCode/post.pkl"

# Open the pickle file in read mode
with open(pickle_file_path, 'rb') as file:
   loaded_data = pickle.load(file)


print(loaded_data)