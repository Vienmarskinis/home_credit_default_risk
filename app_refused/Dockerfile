FROM python:3.11.1-slim

WORKDIR /models

COPY "models/refused_model_V1.joblib" .
COPY "models/refused_preprocessor_V1.joblib" .

WORKDIR /api

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY "app.py" .

EXPOSE 80

CMD ["uvicorn", "app:app","--host", "0.0.0.0","--port", "80"]