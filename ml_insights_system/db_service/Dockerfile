FROM python:3.12-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==2.1.1

# Copy poetry configuration files
COPY pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction

# Copy the application
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "src.db_service.app"]
