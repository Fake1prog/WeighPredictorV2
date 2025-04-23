FROM jupyter/scipy-notebook:python-3.10

USER root
WORKDIR /app

# Install additional system dependencies first
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# Use --no-cache-dir to keep image size down
# Use --default-timeout=300 or higher if needed for slow downloads
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=300

# Now copy the rest of the application code
COPY . .

# Prepare directories (can often be done after copying code)
RUN mkdir -p static/uploads static/results model_output

# Grant ownership to the default jovyan user if needed, although running as root simplifies permissions
# RUN chown -R ${NB_USER}:${NB_GID} /app
# USER ${NB_UID} # Optional: Switch back to non-root user

# Command to run the application
# Ensure $PORT environment variable is set by your deployment environment (like Cloud Run, Heroku etc.)
# If running locally or without PORT set, Gunicorn might default or fail. Provide a default if needed.
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 300 wsgi:app