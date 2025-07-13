import numpy as np
import pickle
from src.som.config import load_train_config

def load_som_weights():

    """
    Load the Self-Organizing Map (SOM) weights from a specified path.

    """

    config = load_train_config()
    weights_path = config["artifacts"]["weights_path"]

    return np.load(weights_path)


