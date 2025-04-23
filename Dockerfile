FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements but don't install everything at once
COPY requirements.txt .

# Install packages in specific order with exact versions
# Install NumPy 1.23.5 FIRST, before anything else
RUN pip install --no-cache-dir numpy==1.23.5 \
    && pip install --no-cache-dir scikit-learn==1.3.0 \
    && pip install --no-cache-dir torch torchvision \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY . .

# Prepare directories
RUN mkdir -p static/uploads static/results model_output

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 wsgi:app