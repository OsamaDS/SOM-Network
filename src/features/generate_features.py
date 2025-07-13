import pandas as pd
import pickle
import traceback
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_features(df: pd.DataFrame, feature_columns: list, scaler_save_path=None) -> np.ndarray:

    """
    
    Generate features from the DataFrame by scaling specified columns.
    
    """

    try:

        scaler = MinMaxScaler()
        features = scaler.fit_transform(df[feature_columns])

        if scaler_save_path:
            with open(scaler_save_path, "wb") as f:
                pickle.dump(scaler, f)

        return features
    
    except Exception as e:
        print(f"Error occurred while generating features: {e}")
        print(traceback.format_exc())
        

    
