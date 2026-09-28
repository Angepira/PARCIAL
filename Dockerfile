FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema mínimas (a veces necesarias para numpy/scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1) Copia solo requirements primero (mejor cache de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copia código y modelo YA entrenado
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Arranca uvicorn. El modelo se carga al importar src.api
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]