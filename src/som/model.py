import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger()

class SOM:

    def __init__(self, width, height, input_dim, learning_rate=0.5, sigma=None,
                n_max_iterations=1000):

        """
        
        This is a self-organizing map (SOM) model.
        It is a type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional representation of the input data.
        The SOM model is used for clustering and visualization of high-dimensional data.
        
        """

        self.width = width
        self.height = height
        self.input_len = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma or max(width, height) / 2
        self.n_max_iterations = n_max_iterations

        # Initialize weights
        self.weights = np.random.random((width, height, input_dim))

        # Precompute neuron grid
        X, Y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        self.grid = np.stack([X, Y], axis=2)
    
    def train(self, input_data):

        """
        
        Train the model using the SOM algorithm.
        
        """
        logger.info("🧠 1- Training of SOM model started...")
        lambda_ = self.n_max_iterations / np.log(self.sigma)
        alpha0 = self.learning_rate
        sigma0 = max(self.height, self.width) / 2

        for t in range(self.n_max_iterations):
            alpha_t = alpha0 * np.exp(-t / lambda_)
            sigma_t = sigma0 * np.exp(-t / lambda_)

            for vt in input_data:
                diff = self.weights - vt
                distances = np.sum(diff**2, axis=2)
                bmu_index = np.argmin(distances)
                bmu_coords = np.unravel_index(bmu_index, (self.width, self.height))

                d_grid = np.sum((self.grid - bmu_coords)**2, axis=2)
                theta_t = np.exp(-d_grid / (2 * sigma_t**2))[..., np.newaxis]

                self.weights += alpha_t * theta_t * (vt - self.weights)

        logger.info("✅ 1- Training of SOM model Finished...")

    def get_weights(self):

        """
        This method returns the current weights of the SOM model.

        """
        return self.weights

    def save_weights(self, path):

        """

        This method saves the current weights of the SOM model to a specified path.

        """
        np.save(path, self.weights)

    def load_weights(self, path):

        """

        This method loads the weights of the SOM model from a specified path.

        """
        self.weights = np.load(path)

    def evaluate_som_model(self, input_data):

        """
        
        This function is used to evaluate SOM model.
        
        """

        errors = []
        for vector in input_data:
            diff = self.weights - vector
            distances = np.sum(diff**2, axis=2)
            min_dist = np.min(distances)
            errors.append(min_dist)
            
        return np.mean(errors)

