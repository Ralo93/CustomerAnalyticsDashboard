FROM python:3.12-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==1.6.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment in the container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY ./bridge_service ./bridge_service
COPY .env ./

# Expose the port the app will run on
EXPOSE 8005

# Command to run the application
CMD ["uvicorn", "bridge_service.src.bridge_service:app", "--host", "0.0.0.0", "--port", "8005"]