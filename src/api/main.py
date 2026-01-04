from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib


# Global model variable
model = None


#Class defnitionto accet request and to return response where we can restrict the input and output 
# data types
class ModelInfo(BaseModel):
    """ Model information structure."""
    model_name: str
    model_type: str
    model_version: str
    model_features: list[str]
    model_author: str
    model_description: str
    model_f1: float

class PredictRequest(BaseModel):
    """ Prediction request structure."""
    
    no_of_trainings : int
    age: int
    length_of_service: int   
    previous_year_rating: int
    KPI_met: int = Field(..., alias="KPIs_met >80%")
    award_won: int = Field(..., alias="awards_won?")
    avg_training_score: int
    department: str
    region: str
    education: str
    gender: str
    recruitment_channel: str
   

class PredictResponse(BaseModel):
    """ Prediction response structure."""

    prediction: str
 

def load_model():
    global model
    model = joblib.load("../../models/promotion_LR_Imb_model.pkl")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    load_model()
    yield
    print("Shutdown...")

app = FastAPI(lifespan=lifespan)


# ## First hook to load a model
# @app.on_event("startup")
# def startup_event():
#     load_model()
#     print("Model loaded successfully")

## Hook triggered when root url loads  
@app.get("/", tags=["Root"])
def read_root():
    """ Root endpoint returning a information.
    """
    return {
        "mesage": "Welcome to the Sepsis Prediction API. Use the /predict endpoint to get predictions.",
        "version": "1.0.0",
        "author": "Shobana Samyayyah",
        "status": "development environemnt",
        "endpoints":{
            "docs": "/docs",
            "redoc": "/redoc",
            "model_info": "/model_info",
            "predict":"/predict"
        }

    }
@app.get("/model_info")
def get_model_info() -> ModelInfo:
    """" Endpoint to get model information."""
    if model is None:
        return {"error": "Model not loaded"}
    else:
        return ModelInfo(
                model_name="Employee Promotion Prediction Model",
                model_type = "Logistic Regression with hyper tuning",
                model_version = "1.0.0",
                model_features = ['no_of_trainings', 'age', 'previous_year_rating', 'KPIs_met >80%', 'awards_won?', 'avg_training_score', 'department', 'region', 'education', 'gender','recruitment_channel'],
                model_author = "Shobana",
                model_description = "A logistic regression model to predict employee promotion",
                model_f1 = 0.48
            )

@app.post("/predict")
def predict(Input : PredictRequest) -> PredictResponse:
 
        """ Endpoint to get predictions."""
        if model is None:
            return {"error": "Model not loaded"}
        else:
        
            X_input = pd.DataFrame([Input.model_dump(by_alias=True)])
          
            prediction = model.predict(X_input)
            result = "Can bePromoted" if prediction[0]==1 else "Not Eligible for Promotion"
            return PredictResponse(prediction=result)
           



