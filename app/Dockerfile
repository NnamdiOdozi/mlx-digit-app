# Use Python 3.9 which has better compatibility with PyTorch CPU
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies ensuring numpy compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    # Install numpy first to ensure it's available for PyTorch
    pip install --no-cache-dir numpy==1.24.3 && \
    # Then install other requirements
    pip install --no-cache-dir -r requirements.txt && \
    # Verify installations
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')" && \
    # Make sure numpy and torch interop works
    python -c "import torch; import numpy as np; t = torch.ones(3,3); print(t.numpy())"

# Copy application code
COPY ./app/ .

# Expose the port
EXPOSE ${PORT}

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.enableWebsocketCompression=false"]