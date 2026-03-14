FROM python:3.10-slim

# Install system dependencies for building Python packages and connecting to PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command (overridden via docker-compose for other services)
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]