FROM jupyter/scipy-notebook:python-3.10

USER root
WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Flask and other web dependencies
RUN pip install --no-cache-dir flask==3.1.0 flask-cors==5.0.1 gunicorn==21.2.0

# Install CV and AWS dependencies
RUN pip install --no-cache-dir opencv-python boto3 pillow

# Copy application code
COPY . .

# Prepare directories
RUN mkdir -p static/uploads static/results model_output

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 wsgi:app