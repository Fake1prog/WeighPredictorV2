FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN mkdir -p static/uploads static/results
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
ENV SECRET_KEY=your-secret-key
ENV ROBOFLOW_API_KEY=FSX72icxHTJs0335DLkL

CMD gunicorn --bind :$PORT modified_app:app