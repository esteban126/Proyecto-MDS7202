# Proyecto SodAI Drinks - Entrega 2

Este proyecto implementa un pipeline de MLops utilizando Apache Airflow, Docker y Python para automatizar el procesamiento de datos, optimización de modelos y generación de predicciones.

---
## Descripción del DAG y Funcionalidad de Cada Tarea

El DAG `SodaI_Drinks` orquesta el siguiente flujo de trabajo:

1. **Start (`EmptyOperator`)**  
   Marca el inicio del pipeline.

2. **Detección de nuevos datos (`BranchPythonOperator: detect_new_data`)**  
   Decide si existen nuevos datos para procesar.  
   - Si hay nuevos datos, continúa con la lectura y procesamiento.
   - Si no, finaliza el flujo sin predecir.

3. **Lectura de nuevos datos (`PythonOperator: read_new_data`)**  
   Lee y consolida los archivos `.parquet` de nuevos datos, y los mueve a la carpeta de datos históricos.

4. **Preprocesamiento (`PythonOperator: preprocessing`)**  
   Realiza la limpieza, transformación y combinación de datos de clientes, productos y transacciones, generando el dataset de entrenamiento.

5. **Branch para reentrenamiento (`BranchPythonOperator: branch_use_current_params`)**  
   Evalúa si el modelo actual sigue siendo suficientemente bueno (drift).  
   - Si el modelo es adecuado, solo predice.
   - Si detecta drift, reentrena y optimiza el modelo.

6. **Optimización y reentrenamiento (`PythonOperator: optimize_model`)**  
   Utiliza Optuna y MLflow para buscar los mejores hiperparámetros y guardar el mejor pipeline.

7. **Predicción (`PythonOperator: predictions`)**  
   Genera predicciones usando el pipeline entrenado y guarda los resultados.

8. **End (`EmptyOperator`)**  
   Marca el final del pipeline.
---
## Diagrama de Flujo del Pipeline

---
## Representación Visual del DAG


---
## Lógica para Integrar Nuevos Datos, Detectar Drift y Reentrenar el Modelo

### Integración de nuevos datos:

El pipeline monitorea la carpeta dags/new_data/. Si detecta archivos nuevos, los procesa y los mueve a old_data/ para mantener el histórico actualizado.

### Detección de drift:

Antes de predecir, el pipeline compara la distribución de las características de la última semana con la de las semanas anteriores utilizando el método MMDDrift de alibi-detect. Si se detecta drift estadístico (es decir, el test indica que la nueva semana es significativamente diferente a las anteriores según el p-value), se activa el reentrenamiento automático del modelo.


### Reentrenamiento automático:

Si se detecta drift, el pipeline ejecuta la tarea optimize_model, que realiza una búsqueda de hiperparámetros con Optuna y registra los resultados en MLflow. El mejor pipeline se guarda y se utiliza para futuras predicciones.



