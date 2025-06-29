#app.py (backend)
from fastapi import FastAPI, HTTPException
from recommender import recommend_products

app = FastAPI()

@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: int):
    try:
        recommendations = recommend_products(customer_id)
        return {"customer_id": customer_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
