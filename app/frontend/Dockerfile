FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY data_testing.csv . 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]