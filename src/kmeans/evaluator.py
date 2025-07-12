from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_kmeans_model(kmeans_model, som_weights, input_dim):

    """
    
    This function is used to evaluate KMeans model
    
    """
    
    flatten_weights = som_weights.reshape(-1, input_dim)

    score = silhouette_score(flatten_weights, kmeans_model.labels_)

    db_index = davies_bouldin_score(flatten_weights, kmeans_model.labels_)

    return score, db_index