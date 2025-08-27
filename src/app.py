# TODO: Implement FastAPI app for churn inference.
# Endpoints: GET /health, POST /predict
from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
from typing import List
from src.io_schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.features import FeaturePreprocessor

app = FastAPI(title="Churn Prediction API", version="1.0.0")

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    model_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_path = os.path.join('artifacts', 'feature_pipeline.pkl')
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model artifacts not found.")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

@app.on_event("startup")
async def startup_event():
    try:
        load_artifacts()
    except Exception as e:
        print(f"Warning: Could not load artifacts: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if model is None or preprocessor is None:
        try:
            load_artifacts()
        except:
            return HealthResponse(status="error")
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or preprocessor is None:
        try:
            load_artifacts()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {e}")
    try:
        data = [item.dict() for item in request.data]
        df = pd.DataFrame(data)
        X_processed = preprocessor.transform(df)  # Puede lanzar error para categorías desconocidas
        probs = model.predict_proba(X_processed)[:, 1]
        predictions = [{"churn_probability": float(p), "churned": bool(p >= 0.5)} for p in probs]
        return PredictionResponse(predictions=predictions)
    except ValueError as ve:
        # Categorías desconocidas u otros errores en preprocesamiento
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
