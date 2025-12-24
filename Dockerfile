# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for OpenCV / YOLO
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

COPY smart_drug_assistant/routers/*.pt smart_drug_assistant/routers/


# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p patient_leaflets chroma_db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/check-db || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "smart_drug_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
