FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install basic dependencies first with compatible versions
RUN pip install --no-cache-dir numpy==1.23.5 \
    && pip install --no-cache-dir scikit-learn==1.3.0 \
    && pip install --no-cache-dir Werkzeug==3.1.0 \
    && pip install --no-cache-dir flask==3.1.0 \
    && pip install --no-cache-dir gunicorn==21.2.0

# Install remaining packages from requirements, skipping already installed ones
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application code
COPY . .

# Prepare directories
RUN mkdir -p static/uploads static/results model_output

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 wsgi:app