#!/usr/bin/env python3
import os
import shutil
import subprocess

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content):
    """Create a file with specified content."""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_project_structure():
    """Create the ML insights system project structure."""
    project_name = "ml-insights-system"
    
    # Remove existing project if it exists
    if os.path.exists(project_name):
        shutil.rmtree(project_name)
        print(f"Removed existing {project_name} directory")
    
    # Create base project directory
    create_directory(project_name)
    
    # Define services
    services = [
        "api_gateway",
        "db_service",
        "queue_service",
        "worker_service",
        "dashboard_service"
    ]
    
    # Create service directories
    for service in services:
        service_dir = os.path.join(project_name, service)
        create_directory(service_dir)
        create_directory(os.path.join(service_dir, "src", service.replace("-", "_")))
        
        # Create Dockerfile
        dockerfile_content = f"""FROM python:3.12-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==2.1.1

# Copy poetry configuration files
COPY pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false \\
    && poetry install --no-interaction

# Copy the application
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "src.{service.replace('-', '_')}.app"]
"""
        create_file(os.path.join(service_dir, "Dockerfile"), dockerfile_content)
        
        # Create pyproject.toml
        pyproject_content = f"""[tool.poetry]
name = "{service.replace('_', '-')}"
version = "0.1.0"
description = "{service.replace('_', ' ').title()} for ML Insights System"
authors = ["Your Name <your.email@example.com>"]
packages = [{{include = "{service.replace('-', '_')}", from = "src"}}]

[tool.poetry.dependencies]
python = "^3.12"
fastapi = "^0.104.0"
uvicorn = "^0.23.2"
httpx = "^0.25.0"
python-dotenv = "^1.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
        create_file(os.path.join(service_dir, "pyproject.toml"), pyproject_content)

    # Create API Gateway specific files
    create_file(
        os.path.join(project_name, "api_gateway", "src", "api_gateway", "app.py"),
        """import os
from typing import Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()
app = FastAPI(title="ML Insights API Gateway")

DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://db_service:8001")
QUEUE_SERVICE_URL = os.getenv("QUEUE_SERVICE_URL", "http://queue_service:8002")

class SentenceInput(BaseModel):
    text: str

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/insights/sentences")
async def process_sentence(sentence: SentenceInput) -> Dict[str, str]:
    try:
        # Store in database
        async with httpx.AsyncClient() as client:
            db_response = await client.post(
                f"{DB_SERVICE_URL}/sentences", 
                json={"text": sentence.text}
            )
            
            sentence_id = db_response.json().get("id")
            
            # Queue for processing
            await client.post(
                f"{QUEUE_SERVICE_URL}/queue", 
                json={"sentence_id": sentence_id}
            )
                
            return {
                "id": sentence_id,
                "status": "queued",
            }
            
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Service unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
"""
    )
    
    # Create Nginx config
    create_file(
        os.path.join(project_name, "api_gateway", "nginx.conf"),
        """http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Define upstream servers
    upstream api_servers {
        server api_gateway:8000;
    }
    
    upstream dashboard {
        server dashboard_service:8501;
    }
    
    server {
        listen 80;
        
        # Health check
        location /health {
            proxy_pass http://api_servers/health;
        }
        
        # API Gateway
        location /api/ {
            rewrite ^/api/(.*) /$1 break;
            proxy_pass http://api_servers;
        }
        
        # Streamlit Dashboard
        location / {
            proxy_pass http://dashboard;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}"""
    )

    # Create docker-compose.yml
    create_file(
        os.path.join(project_name, "docker-compose.yml"),
        """version: '3.8'

services:
  # Nginx API Gateway
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./api_gateway/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api_gateway
    networks:
      - ml_insights_network

  # API Gateway Service
  api_gateway:
    build:
      context: ./api_gateway
    env_file:
      - .env
    environment:
      - DB_SERVICE_URL=http://db_service:8001
      - QUEUE_SERVICE_URL=http://queue_service:8002
    networks:
      - ml_insights_network

  # Database Service
  db_service:
    build:
      context: ./db_service
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ml_insights
    depends_on:
      - postgres
    networks:
      - ml_insights_network

  # Queue Service
  queue_service:
    build:
      context: ./queue_service
    environment:
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
    depends_on:
      - rabbitmq
    networks:
      - ml_insights_network

  # Worker Service
  worker_service:
    build:
      context: ./worker_service
    environment:
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ml_insights
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - rabbitmq
      - postgres
    networks:
      - ml_insights_network

  # PostgreSQL Database
  postgres:
    image: postgres:14
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=ml_insights
    networks:
      - ml_insights_network

  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3-management
    networks:
      - ml_insights_network

networks:
  ml_insights_network:"""
    )

    # Create .env.example
    create_file(
        os.path.join(project_name, ".env.example"),
        """# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ml_insights
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ml_insights

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Service URLs
DB_SERVICE_URL=http://db_service:8001
QUEUE_SERVICE_URL=http://queue_service:8002"""
    )

    # Create minimal empty files for other services
        # Update db_service pyproject.toml to include SQLAlchemy
    db_pyproject_content = """[tool.poetry]
name = "db-service"
version = "0.1.0"
description = "Database Service for ML Insights System"
authors = ["Your Name <your.email@example.com>"]
packages = [{include = "db_service", from = "src"}]

[tool.poetry.dependencies]
python = "^3.12"
fastapi = "^0.104.0"
uvicorn = "^0.23.2"
httpx = "^0.25.0"
python-dotenv = "^1.0.0"
sqlalchemy = "^2.0.23"
psycopg2-binary = "^2.9.9"
alembic = "^1.12.1"
pydantic = "^2.4.2"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
    create_file(os.path.join(project_name, "db_service", "pyproject.toml"), db_pyproject_content)
    
    # Create DB models file
    create_file(
        os.path.join(project_name, "db_service", "src", "db_service", "models.py"),
        """from datetime import datetime
import uuid
from sqlalchemy import Column, String, Text, DateTime, Enum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import enum

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/ml_insights")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PriorityLevel(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class Sentence(Base):
    __tablename__ = "sentences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(Text, nullable=False)
    priority = Column(String, default="normal")
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    insights = Column(Text, nullable=True)

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)
"""
    )
    
    create_file(
        os.path.join(project_name, "db_service", "src", "db_service", "service.py"),
        """from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
from . import models
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Database Service")

# Dependency to get DB session
def get_db():
    db = models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SentenceCreate(BaseModel):
    text: str
    priority: Optional[str] = "normal"

class SentenceResponse(BaseModel):
    id: str
    text: str
    priority: str

@app.on_event("startup")
async def startup_event():
    models.create_tables()

@app.post("/sentences", response_model=SentenceResponse, status_code=201)
async def create_sentence(sentence: SentenceCreate, db: Session = Depends(get_db)):
    db_sentence = models.Sentence(
        id=str(uuid.uuid4()),
        text=sentence.text,
        priority=sentence.priority
    )
    db.add(db_sentence)
    db.commit()
    db.refresh(db_sentence)
    return db_sentence

@app.get("/sentences/{sentence_id}")
async def get_sentence(sentence_id: str, db: Session = Depends(get_db)):
    sentence = db.query(models.Sentence).filter(models.Sentence.id == sentence_id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")
    return sentence

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8001, reload=True)
"""
    )
    
    create_file(
        os.path.join(project_name, "queue_service", "src", "queue_service", "service.py"),
        """from fastapi import FastAPI

app = FastAPI(title="Queue Service")

@app.post("/queue")
async def queue_task(task_data: dict):
    # Placeholder: publish to RabbitMQ
    return {"status": "queued"}
"""
    )
    
    create_file(
        os.path.join(project_name, "worker_service", "src", "worker_service", "worker.py"),
        """import os
import time

def process_sentence(sentence_id):
    # Placeholder: process sentence using OpenAI API
    print(f"Processing sentence {sentence_id}")
    
if __name__ == "__main__":
    # Simple polling loop for demo purposes
    while True:
        print("Worker checking for messages...")
        time.sleep(10)
"""
    )
    
    create_file(
        os.path.join(project_name, "dashboard_service", "src", "dashboard_service", "app.py"),
        """import streamlit as st

st.title("ML Insights Dashboard")

st.write("This dashboard will display insights from processed sentences.")

# Placeholder data
st.metric(label="Processed Sentences", value="0")
"""
    )
    
    # Create __init__.py files
    for service in services:
        init_file = os.path.join(project_name, service, "src", service.replace("-", "_"), "__init__.py")
        create_file(init_file, "")
        src_init = os.path.join(project_name, service, "src", "__init__.py")
        create_file(src_init, "")

    print(f"\nProject {project_name} created successfully!")
    print("To get started:")
    print(f"  cd {project_name}")
    print("  cp .env.example .env  # Update with your API keys")
    print("  docker-compose up -d")

def print_local_dev_instructions():
    """Print instructions for local development."""
    import platform
    is_windows = platform.system() == "Windows"
    
    print("\n=== Local Development Instructions ===")
    print("To verify services locally before containerization:")
    
    print("\n1. Install Poetry globally if not already installed:")
    print("   pip install poetry==2.1.1")
    
    print("\n2. For each service (replace SERVICE_NAME as needed):")
    print("   cd ml-insights-system/SERVICE_NAME")
    print("   poetry install")
    print("   poetry shell")
    
    print("\n3. Run each service locally (in separate terminals):")
    
    if is_windows:
        print("\n   Windows Command Prompt:")
        print("   # Database service (set up PostgreSQL locally first)")
        print("   cd ml-insights-system\\db_service")
        print("   SET DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ml_insights")
        print("   poetry run python -m src.db_service.service")
        
        print("\n   # Queue service")
        print("   cd ml-insights-system\\queue_service")
        print("   poetry run python -m src.queue_service.service")
        
        print("\n   # API Gateway")
        print("   cd ml-insights-system\\api_gateway")
        print("   SET DB_SERVICE_URL=http://localhost:8001")
        print("   SET QUEUE_SERVICE_URL=http://localhost:8002")
        print("   poetry run python -m src.api_gateway.app")
        
        print("\n   Windows PowerShell:")
        print("   # Database service")
        print("   cd ml-insights-system\\db_service")
        print('   $env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/ml_insights"')
        print("   poetry run python -m src.db_service.service")
        
        print("\n   # API Gateway")
        print("   cd ml-insights-system\\api_gateway")
        print('   $env:DB_SERVICE_URL = "http://localhost:8001"')
        print('   $env:QUEUE_SERVICE_URL = "http://localhost:8002"')
        print("   poetry run python -m src.api_gateway.app")
    else:
        print("   # Database service (set up PostgreSQL locally first)")
        print("   cd ml-insights-system/db_service")
        print("   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ml_insights")
        print("   poetry run python -m src.db_service.service")
        
        print("   # Queue service")
        print("   cd ml-insights-system/queue_service")
        print("   poetry run python -m src.queue_service.service")
        
        print("   # API Gateway")
        print("   cd ml-insights-system/api_gateway")
        print("   export DB_SERVICE_URL=http://localhost:8001")
        print("   export QUEUE_SERVICE_URL=http://localhost:8002")
        print("   poetry run python -m src.api_gateway.app")
    
    print("\n4. Once verified, run with Docker:")
    print("   cp .env.example .env  # Update with your API keys")
    print("   docker-compose up -d")
    
    print("\n5. Testing with curl (or use Postman/Insomnia):")
    print('   curl -X POST http://localhost:8000/insights/sentences -H "Content-Type: application/json" -d \'{"text": "This is a test sentence"}\'')

if __name__ == "__main__":
    create_project_structure()
    print_local_dev_instructions()