FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY agent.py .
COPY university_mental_health_iot_dataset.csv .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PGHOST=postgres
ENV PGDATABASE=users
ENV PGUSER=postgres
ENV PGPASSWORD=example
ENV PGPORT=5432
ENV CSV_PATH=/app/university_mental_health_iot_dataset.csv

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

