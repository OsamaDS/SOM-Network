from pydantic import BaseModel

class PredictRequest(BaseModel):
    
    age: float
    income: float
    spending_score: float
