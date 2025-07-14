from src.kmeans.evaluator import evaluate_kmeans_model
from src.som.config import load_train_config
import json

config = load_train_config()

def evaluate_model(som_model, input_data, cluster_model):
    
    """

    Evaluate SOM and optionally KMeans clustering.
    
    """
    # Quantization Error
    mean_error = som_model.evaluate_som_model(input_data)
    som_weights = som_model.get_weights()
    input_dim = config['som-model']['input_dim']
    kmeans_model = cluster_model.get_model()

    score, db_index = evaluate_kmeans_model(kmeans_model, som_weights, input_dim)

    results = {
        "quantization_error": mean_error
    }

    if kmeans_model is not None:
    
        results["kmeans_inertia"] = kmeans_model.inertia_
        results["silhouette_score"] = score
        results["davies_bouldin"] = db_index

    with open(config['artifacts']['metrics_path'], "w") as f:
            json.dump(results, f, indent=2)
            
    return results