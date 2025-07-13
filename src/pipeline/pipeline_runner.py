from src.pipeline.data_ingestion import load_data
from src.pipeline.data_validation import validate_data
from src.pipeline.feature_engineering import build_feature
from src.pipeline.model_trainer import train_model
from src.pipeline.model_evaluator import evaluate_model
from src.som.config import load_train_config
from src.utils.logger import get_logger
from src.utils.utils import load_model
import numpy as np
import traceback

logger = get_logger()

def run_pipeline():

    """
    
    This function runs the entir pipeline for training of the model
    
    """

    try:
        config = load_train_config()

        logger.info("🔄 Starting training pipeline...")

        # Step 1: Ingest
        logger.info("📥 Loading data...")
        df = load_data(config['data']['raw_path'])

        # Step 2: Validate
        logger.info("🔎 Validating data...")
        is_validate = validate_data(df, config['feature-store']['input_columns'])

        if not is_validate:
            raise ValueError(
                f"Data validation failed. Check that all required columns "
                f"{config['feature-store']['input_columns']} are present and valid."
            )
        
        # Step 3: Feature Engineering
        logger.info("🧪 Generating features...")
        features = build_feature(df, config)

        # Step 4: Train model
        logger.info("🧠 Training model...")
        train_model(features, config)
        logger.info("✅ Model trained successfully...")

        som_model = load_model(config['artifacts']['model_path'])
        kmeans_model = load_model(config['artifacts']['kmeans_model_path'])
        input_data = input_data = np.load(config['feature-store']['output_path'])

        logger.info("📈 Evaluating model...")
        metrics = evaluate_model(som_model, input_data, kmeans_model)
        logger.info(f"Model metrics: {metrics}")




    except Exception as e:
        print(f"Error occur while running the pipeline function: {e}")
        traceback.print_exc()
