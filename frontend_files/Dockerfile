# Use a lightweight Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install the dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the Streamlit app
COPY app.py .

# Default backend address = backend container name on the shared Docker network
ENV BACKEND_URL=http://backend:7860

# Streamlit runs on port 8501
EXPOSE 8501

# Start the Streamlit app, listening on all interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
