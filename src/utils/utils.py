import pickle

def save_model(model, path):

    """

    Save the entire SOM model object to disk as a pickle file.

    """

    model_path = path

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to path: {model_path}")

def load_model(path):

    """

    Load the SOM model object from disk.
    
    """

    model_path = path

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

