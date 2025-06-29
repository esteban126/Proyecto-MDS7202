# recommender.py

import pandas as pd

# Cargar dataset real de historial de compras
df = pd.read_parquet("transacciones.parquet")

# Obtener los 5 productos más populares
def top_products(n=5):
    return df.groupby("product_id")["items"].sum().sort_values(ascending=False).head(n).index.tolist()

# Recomendador basado en los productos más comprados por el cliente
def recommend_products(customer_id, n=5):
    if customer_id not in df["customer_id"].unique():
        return top_products(n)

    # Productos más comprados por el cliente
    compras_cliente = (
        df[df["customer_id"] == customer_id]
        .groupby("product_id")["items"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Completar si hay menos de 5 productos
    if len(compras_cliente) < n:
        adicionales = [p for p in top_products(n=10) if p not in compras_cliente]
        compras_cliente += adicionales[: n - len(compras_cliente)]

    return compras_cliente[:n]
