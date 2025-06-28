# Use a base Python image
FROM python:3.9-slim

# Set working directory and Airflow home
WORKDIR /root/airflow

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Airflow with constraints
RUN pip install --no-cache-dir "apache-airflow==2.7.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.9.txt"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DAGs and utils to Airflow home
COPY dags/ /root/airflow/dags/
COPY utils/ /root/airflow/utils/
COPY logs/ /root/airflow/logs/

# Initialize Airflow database
RUN airflow db init

# Create admin user
RUN airflow users create \
    --role Admin \
    --username admin \
    --email admin@example.com \
    --firstname admin \
    --lastname admin \
    --password admin

# Expose Airflow webserver port
EXPOSE 8080

# Comando para iniciar el servidor web y el scheduler
CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]