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

# Install packages in specific order with exact versions
RUN pip install --no-cache-dir numpy==1.23.5 \
    && pip install --no-cache-dir scikit-learn==1.3.0 \
    && pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 \
    && pip install --no-cache-dir gunicorn==21.2.0 \
    && pip install --no-cache-dir opencv-python-headless==4.8.0.74 \
    && pip install --no-cache-dir matplotlib==3.7.2 \
    && pip install --no-cache-dir flask==3.1.0 flask-cors==5.0.1 Werkzeug==3.0.1 \
    && pip install --no-cache-dir boto3==1.18.0 pillow==10.0.0 tqdm==4.66.1

# Copy application code
COPY . .

# Prepare directories
RUN mkdir -p static/uploads static/results model_output

# Command to run the application - use the full ML app
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 wsgi:app