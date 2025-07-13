import traceback
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field
from src.predictor.predict import predict_cluster
from src.api.schema import PredictRequest

router = APIRouter()

@router.post("/predict")
async def predict_route(request: PredictRequest):

    """

    Asynchronously predict the cluster for new customer data.

    """

    try:
        new_data = np.array([[request.age, request.income, request.spending_score]])
        prediction = predict_cluster(new_data) 

        return {"cluster": prediction}
    
    except Exception as e:
        print("Error occured while calling the predict end point: {e}")
        traceback.print_exc()
