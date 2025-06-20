FROM apache/airflow:2.8.1-python3.10
USER root
COPY requirements.txt /
RUN pip install --upgrade pip && pip install -r /requirements.txt
USER airflow
