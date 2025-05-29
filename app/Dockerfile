# Build stage
FROM python:3.12.4-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Force CPU-only installation and exclude NVIDIA components
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt \
    # Remove NVIDIA and CUDA components
    && rm -rf /usr/local/lib/python3.12/site-packages/nvidia \
    && rm -rf /usr/local/lib/python3.12/site-packages/triton \
    && rm -rf /usr/local/lib/python3.12/site-packages/cusparselt \
    && rm -rf /usr/local/lib/python3.12/site-packages/*cuda* \
    # Clean up
    && find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.12/site-packages -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Final stage
FROM python:3.12.4-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy application code
COPY ./app/ .

EXPOSE ${PORT}

CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false"]