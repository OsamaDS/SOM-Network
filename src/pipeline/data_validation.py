import pandas as pd

def validate_data(df: pd.DataFrame, required_columns: list) -> bool:

    """

    Validate the DataFrame to ensure it contains the required columns.
    
    """

    if not isinstance(df, pd.DataFrame):
        print("Error: The input is not a pandas DataFrame.")
        return False

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing: {missing_columns}")
        return False
    
    if df[required_columns].isnull().values.any():
        print("Error: The DataFrame contains null values in required columns.")
        return False

    print("Data validation passed.")
    return True