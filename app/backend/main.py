from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import utils

# Rutas absolutas a los archivos montados desde airflow
MODEL_PATH = "/app/airflow/models/full_pipeline.pkl"
DATA_PATH = "/app/airflow/models/data_testing.csv"

# Cargar modelo y datos
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

data_df = pd.read_csv(DATA_PATH)


app = FastAPI(title="SodAI Drinks Predictor")

from typing import List, Optional

class QueryInput(BaseModel):
    customer_id: str
    brands: Optional[List[str]] = None
    packages: Optional[List[str]] = None

@app.post("/predict")
def predict(data: QueryInput):
    try:
        filtered = data_df[data_df["customer_id"].astype(str) == data.customer_id]
        if data.brands:
            filtered = filtered[filtered["brand"].isin(data.brands)]
        if data.packages:
            filtered = filtered[filtered["package"].isin(data.packages)]
        if filtered.empty:
            return {"customer_id": data.customer_id, "recommended_products": []}
        preds = pipeline.predict(filtered)
        filtered = filtered.copy()
        filtered["prediction"] = preds
        products = filtered.loc[filtered["prediction"] == 1, "product_id"].tolist()
        return {"customer_id": data.customer_id, "recommended_products": products}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))