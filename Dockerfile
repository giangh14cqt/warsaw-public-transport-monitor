FROM python:3.12-slim

# Set timezone, disable bytecode generation, and ensure unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Warsaw

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY collector_daemon.py .
COPY run_preflight.py .

# Create volume directories
RUN mkdir -p data/raw data/gtfs logs

VOLUME ["/app/data", "/app/logs"]

# Launch daemon by default
CMD ["python3", "collector_daemon.py"]
