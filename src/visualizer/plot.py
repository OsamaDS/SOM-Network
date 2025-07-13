import numpy as np
import traceback
import matplotlib.pyplot as plt

def save_plot_som_grid(som_weights: np.ndarray, save_path: str, dpi=300):

    """
    
    Save SOM plot.
    
    """

    try:
        image = som_weights / np.max(som_weights)

        plt.figure(figsize=(6, 6))
        plt.imshow(image, aspect='auto')
        plt.title("SOM Grid Visualization (15x15)")
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close() 
        
        print(f"Plot saved to: {save_path}")


    except Exception as e:
        print(f"Error while saving the SOM grid plot: {e}")
        traceback.print_exc()

    
def save_cluster_plot(cluster_map: list, save_path: str, dpi=300):

    """

    Save Cluster plot 
    
    """

    try:

        plt.figure(figsize=(8, 8))
        plt.imshow(cluster_map, cmap='Accent', interpolation='nearest')
        plt.title("SOM Grid with KMeans Cluster Labels")
        plt.colorbar(label='Cluster Label')
        plt.xlabel("Neuron X")
        plt.ylabel("Neuron Y")
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close() 

    except Exception as e:
        print('Error occured while saving the cluster plot: {e}')
        traceback.print_exc()