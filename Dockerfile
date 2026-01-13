FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by dlib and face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

<<<<<<< HEAD
# Expose port
EXPOSE 8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "eventlet", "-w", "1", "--timeout", "120", "server:app"]
=======
# Expose port (Railway will set PORT env variable)
EXPOSE 8080

# Run the application - Railway provides PORT env variable
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --worker-class eventlet -w 1 --timeout 120 server:app
>>>>>>> 0118eb0 (Use libgl1 and PORT env var in Dockerfile; use  in nixpacks startCommand)
