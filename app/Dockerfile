# Dockerfile
# # ---- Stage 1: Build stage ----
FROM python:3.12.4-slim AS build

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install build-time system dependencies (needed to *install* Python packages)
# git/curl might only be needed if requirements.txt pulls directly from git repos etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir helps keep this layer smaller if intermediate layers are cached
# The retry loop is good practice for network issues
RUN pip install --upgrade pip \
    && bash -c 'for i in {1..5}; do pip install --no-cache-dir -r requirements.txt && break || (echo "Install failed... retrying in 10s" && sleep 10); done'

# ---- Stage 2: Production stage ----
FROM python:3.12.4-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install runtime system dependencies (needed to *run* the application)
# libpq-dev contains runtime libraries for postgresql connection (like libpq.so.5)
# --no-install-recommends keeps the image smaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    # Add any other *runtime* system dependencies here
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the build stage
# Adjust the python version path if you change the base image
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# Copy executables installed by pip (like streamlit)
COPY --from=build /usr/local/bin /usr/local/bin

# Copy the application code from the build context (your local machine)
# This assumes main.py and other necessary files are in the same directory as the Dockerfile
COPY . .
# If your code is in a subdirectory (e.g., 'src'), use:
# COPY ./src .

EXPOSE ${PORT}

# Use exec form (JSON array) for CMD is generally preferred
# Ensures PORT variable is evaluated by the shell running the container, not at build time
CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0"]

# Alternative CMD using direct execution (if PORT doesn't need shell evaluation, but it does here)
# CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]