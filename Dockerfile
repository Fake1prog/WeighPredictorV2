FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and build tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Prepare directories
RUN mkdir -p static/uploads static/results model_output

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 wsgi:app