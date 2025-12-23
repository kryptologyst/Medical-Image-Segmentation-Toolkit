# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints data evaluation_results assets

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD python -c "import torch; print('PyTorch available')" || exit 1

# Default command
CMD ["streamlit", "run", "demo/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
