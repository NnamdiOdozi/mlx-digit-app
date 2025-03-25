# Dockerfile
# ---- Base Python image ----
    FROM python:3.12.4-slim

    # ---- Set environment variables ----
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PORT=8501 \
        DEBIAN_FRONTEND=noninteractive
    
    # ---- Set working directory ----
    WORKDIR /app
    
    # ---- System dependencies ----
    RUN apt-get update && apt-get install -y \
        build-essential \
        libpq-dev \
        curl \
        git \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Install Python dependencies ----
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip config set global.index-url https://pypi.python.org/simple \
    && bash -c 'for i in {1..5}; do pip install --no-cache-dir -r requirements.txt && break || (echo "Install failed... retrying in 10s" && sleep 10); done'

    # ---- Copy the entire app ----
    COPY . .
    
    # ---- Expose the dynamic port ----
    EXPOSE ${PORT}
    
    # ---- Run the Streamlit app ----
        CMD streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
