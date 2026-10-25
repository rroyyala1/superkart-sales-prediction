# Use a lightweight Python image (same minor version as my training environment)
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install the dependencies first (Docker caches this layer between builds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the Flask app and the serialized model
COPY app.py superkart_model.joblib ./

# The API listens on port 7860
EXPOSE 7860

# Start the app with gunicorn: 4 workers, bound to port 7860 -> <module>:<Flask app object>
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "app:superkart_api"]
