# app.py (Frontend - Gradio)

import gradio as gr
import requests

def obtener_recomendaciones(cliente_id):
    try:
        url = f"http://backend:8000/recommend/{int(cliente_id)}"
        response = requests.get(url)
        if response.status_code == 200:
            recs = response.json()["recommendations"]
            return f"Recomendaciones para el cliente {cliente_id}:", recs
        else:
            return f"Error: {response.status_code} - {response.text}", []
    except Exception as e:
        return f"Error de conexión: {str(e)}", []

iface = gr.Interface(
    fn=obtener_recomendaciones,
    inputs=gr.Textbox(label="Ingrese ID del cliente"),
    outputs=[gr.Text(label="Estado"), gr.JSON(label="Recomendaciones")],
    title="SodAI Drinks Recomendador de Productos"
)

iface.launch(server_name="0.0.0.0", server_port=8501)
