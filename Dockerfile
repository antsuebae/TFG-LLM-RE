FROM python:3.12-slim

# Dependencias del sistema necesarias para numpy/scipy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python primero (capa cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y assets
COPY src/    ./src/
COPY config/ ./config/
COPY data/   ./data/

# Crear directorio de resultados
RUN mkdir -p results/experiments results/analysis results/logs results/pipeline results/checkpoints

# El cliente ollama respeta OLLAMA_HOST; se sobreescribe en docker-compose
ENV OLLAMA_HOST=http://ollama:11434
ENV PYTHONPATH=/app/src

# Comando por defecto: Streamlit (sobreescrito por docker-compose para la API)
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
