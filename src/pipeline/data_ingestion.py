import pandas as pd
import traceback

def load_data(path: str) -> pd.DataFrame:

    """
    
    Load data from a CSV file into a pandas DataFrame.
    
    """

    try:

        data = pd.read_csv(path)

        return data
    
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        traceback.print_exc()
