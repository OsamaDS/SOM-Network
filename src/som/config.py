import os
import yaml

TRAIN_CONFIG_PATH = os.path.join("configs", "train_config.yaml")

def load_train_config():

    """
    
    Load training configuration from a YAML file.
    
    """
    
    with open(TRAIN_CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config

