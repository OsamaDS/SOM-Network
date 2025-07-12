from src.features.generate_features import generate_features
import pandas as pd
import numpy as np
import traceback
import json

def build_feature(df: pd.DataFrame, config: dict) -> np.ndarray:

    """

    Build features from the DataFrame based on the configuration provided.

    """

    try:
        cols = config['feature-store']['input_columns']
        scaler_type = config['feature-store']['scaler']
        output_path = config['feature-store']['output_path']
        registry_path = config['feature-store']['registry_path']
        scaler_model_path = config['feature-store']['scaler_model_path']

        features = generate_features(df, cols, scaler_model_path)
        np.save(output_path, features)

        # Save metadata to registry
        registry = {
            "columns": cols,
            "scaler": scaler_type,
            "output_path": output_path,
            "scaler_path": scaler_model_path
        }
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        return features


    except Exception as e:
        print(f"Error occurred while building features: {e}")
        traceback.print_exc()