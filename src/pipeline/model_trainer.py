import numpy as np
import traceback
from src.som.model import SOM
from src.kmeans.cluster import KMEANS
from src.utils.utils import save_model
from src.visualizer.plot import save_plot_som_grid, save_cluster_plot

def train_model(features: np.ndarray, config: dict):

    """

    Train the Self-Organizing Map (SOM) model using the provided configuration.

    """

    try:
        width, height = config['som-model']['grid_size']
        input_dim = features.shape[1]
        sigma = config['som-model']['sigma']
        learning_rate = config['som-model']['learning_rate']
        num_iterations = config['som-model']['num_iterations']
        save_som_plot_path = config['artifacts']['save_som_plot']

        n_clusters = config['kmeans-model']['n_clusters']
        n_init = config['kmeans-model']['init']
        save_cluster_path = config['artifacts']['save_cluster_plot']

        som = SOM(width, height, input_dim, learning_rate, sigma, num_iterations)
        som.train(features)
        som.save_weights(config['artifacts']['weights_path'])
        save_model(som, config['artifacts']['model_path'])

        #saving the SOM plot
        som_weights = som.get_weights()
        save_plot_som_grid(som_weights, save_som_plot_path)

        kmeans = KMEANS(som_weights, width, height, input_dim, n_clusters, n_init)
        cluster_map = kmeans.train()
        save_model(kmeans, config['artifacts']['kmeans_model_path'])
        save_cluster_plot(cluster_map, save_cluster_path)






    except Exception as e:
        print(f"Error occurred while training the model: {e}")
        traceback.print_exc()

