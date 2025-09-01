# File location: docker/Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash jax-user
USER jax-user

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["python", "-c", "import jax; print('JAX-NSL container ready!')"]