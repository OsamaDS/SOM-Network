from sklearn.cluster import KMeans
import numpy as np
import traceback
from src.utils.logger import get_logger

logger = get_logger()

class KMEANS:

    def __init__(self, som_weights, width, height, input_dim, n_clusters, n_init=10):

        self.som_weights = som_weights
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.width = width
        self.height = height

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=self.n_init)

    def train(self):

        """
        
        This function trains the Kmeans clustering model.
        
        """

        try:

            logger.info("🧠 2- Training of cluster model started...")
            flattened_weights = self.som_weights.reshape(-1, self.input_dim)
            cluster_labels = self.kmeans.fit_predict(flattened_weights)
            # Step 3: Reshape labels back to SOM grid shape
            cluster_map = cluster_labels.reshape(self.width, self.height)
            logger.info("✅ 2- Training of cluster model Finished...")

            return cluster_map

        except Exception as e:
            print(f"Error occured while training the KMeans model: {e}")
            traceback.print_exc()


