import gradio as gr
import requests
import pandas as pd

API_URL = "http://sodai_backend:8000/predict"

data_df = pd.read_csv("data_testing.csv")
brands = sorted(data_df["brand"].astype(str).unique())
packages = sorted(data_df["package"].astype(str).unique())

def get_products(customer_id, selected_brands, selected_packages):
    data = {
        "customer_id": customer_id,
        "brands": selected_brands,
        "packages": selected_packages
    }
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        result = response.json()
        prods = result.get("recommended_products", [])
        if prods:
            return f"Productos recomendados: {', '.join(map(str, prods))}"
        else:
            return "No hay productos recomendados para este cliente."
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# SodAI Drinks 🥤 - Recomendador de Productos")
    gr.Markdown("Introduce el customer_id y filtra por marca y envase.")
    customer_id = gr.Textbox(label="Customer ID")
    brand_select = gr.Dropdown(choices=brands, label="Marca", multiselect=True)
    package_select = gr.Dropdown(choices=packages, label="Envase", multiselect=True)
    btn = gr.Button("Consultar")
    output = gr.Textbox(label="Productos recomendados")

    btn.click(get_products, inputs=[customer_id, brand_select, package_select], outputs=output)
    gr.Markdown("**¿Cómo usar la app?**\n\nIngresa el ID del cliente, selecciona marcas y envases, y presiona 'Consultar'.")

demo.launch(server_name="0.0.0.0", server_port=7860)