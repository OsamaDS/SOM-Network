import numpy as np
import pickle
import traceback
from src.utils.utils import load_model
from src.som.config import load_train_config

def predict_cluster(new_data: np.ndarray):

    """
    
    This function predicts the cluster of a new data point using the trained SOM + KMeans.
    
    """

    try:
        config = load_train_config()

        som_model = load_model(config['artifacts']['model_path'])
        kmeans_model_obj = load_model(config['artifacts']['kmeans_model_path'])
        kmeans_model = kmeans_model_obj.get_model()

        scaler = load_model(config['feature-store']['scaler_model_path'])

        scaled_data = scaler.transform(new_data)
        weights = som_model.get_weights().reshape(-1, scaled_data.shape[1])

        bmu_indices = []
        for vector in scaled_data:
            distances = np.linalg.norm(weights - vector, axis=1)
            bmu_index = np.argmin(distances)
            bmu_indices.append(bmu_index)

        # Predict cluster using BMU weight index
        bmu_weights = weights[bmu_indices]
        cluster_preds = kmeans_model.predict(bmu_weights)

        return cluster_preds.tolist()

    except Exception as e:
        print(f"Error occured while predicting the cluster: {e}")
        traceback.print_exc()