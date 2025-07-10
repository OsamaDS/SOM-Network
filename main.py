import pandas as pd
import numpy as np
from src.som.model import SOM
from src.som.config import DATA_PATH

def main():
    print("Hello from mantel-group!")

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully!")

    print("Initializing SOM model...")
    som = SOM(data=df)


if __name__ == "__main__":
    main()
