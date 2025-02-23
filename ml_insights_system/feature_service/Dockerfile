FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8003

# Expose port
EXPOSE 8003

# Run the service
CMD ["poetry", "run", "uvicorn", "src.feature_service.service:app", "--host", "0.0.0.0", "--port", "8003"]