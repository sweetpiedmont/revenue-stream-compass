# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for WeasyPrint + fonts
RUN apt-get update && apt-get install -y \
    build-essential \
    libpango-1.0-0 \
    libcairo2 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose Cloud Run’s default port
EXPOSE 8080

# Run a lightweight server with Flask to serve mini_report
CMD ["python", "pdf_generator/server.py"]
