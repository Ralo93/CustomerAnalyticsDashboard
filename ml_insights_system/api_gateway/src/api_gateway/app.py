import os
import logging
import time
import uuid
from typing import Dict, Optional
from datetime import datetime
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Service URLs
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
QUEUE_SERVICE_URL = os.getenv("QUEUE_SERVICE_URL", "http://localhost:8002")
FEATURE_SERVICE_URL = os.getenv("FEATURE_SERVICE_URL", "http://localhost:8003")

app = FastAPI(title="ML Insights API Gateway")

class SentenceInput(BaseModel):
    external_id: str
    text: str

class ServiceHealthStatus(BaseModel):
    api_gateway: str
    db_service: str
    feature_service: str
    queue_service: str
    timestamp: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests with timing information"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.4f}s: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Log when the API Gateway starts"""
    logger.info("API Gateway Service starting up")
    logger.info(f"Database Service URL: {DB_SERVICE_URL}")
    logger.info(f"Queue Service URL: {QUEUE_SERVICE_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    """Log when the API Gateway shuts down"""
    logger.info("API Gateway Service shutting down")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check for the API gateway"""
    logger.info("Health check requested")
    return {
        "status": "healthy", 
        "service": "api_gateway",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health/services")
async def services_health_check() -> ServiceHealthStatus:
    """Check health of all services"""
    logger.info("Services health check requested")
    
    health_status = {
        "api_gateway": "healthy",
        "db_service": "unknown",
        "queue_service": "unknown",
        "feature_service": "unknown",  # Add this line
        "timestamp": datetime.now().isoformat()
    }

    # Check database service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            logger.debug(f"Checking database service health at {DB_SERVICE_URL}/health")
            response = await client.get(f"{DB_SERVICE_URL}/health")
            
            if response.status_code == 200:
                health_status["db_service"] = "healthy"
                logger.info("Database service is healthy")
            else:
                health_status["db_service"] = "unhealthy"
                logger.warning(f"Database service returned status code {response.status_code}")
    except httpx.RequestError as e:
        health_status["db_service"] = "unavailable"
        logger.error(f"Failed to connect to database service: {str(e)}")

    # Check queue service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            logger.debug(f"Checking queue service health at {QUEUE_SERVICE_URL}/health")
            response = await client.get(f"{QUEUE_SERVICE_URL}/health")
            
            if response.status_code == 200:
                health_status["queue_service"] = "healthy"
                logger.info("Queue service is healthy")
            else:
                health_status["queue_service"] = "unhealthy"
                logger.warning(f"Queue service returned status code {response.status_code}")
    except httpx.RequestError as e:
        health_status["queue_service"] = "unavailable"
        logger.error(f"Failed to connect to queue service: {str(e)}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            logger.debug(f"Checking feature service health at {FEATURE_SERVICE_URL}/health")
            response = await client.get(f"{FEATURE_SERVICE_URL}/health")
            
            if response.status_code == 200:
                health_status["feature_service"] = "healthy"
                logger.info("Feature service is healthy")
            else:
                health_status["feature_service"] = "unhealthy"
                logger.warning(f"Feature service returned status code {response.status_code}")
    except httpx.RequestError as e:
        health_status["feature_service"] = "unavailable"
        logger.error(f"Failed to connect to feature service: {str(e)}")

    return health_status

@app.post("/sentences", status_code=201)
async def process_sentence(sentence: SentenceInput, response: Response) -> Dict[str, str]:
    """
    Process a new sentence:
    1. Store in database
    2. Extract features
    3. Queue for processing
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing new sentence request: {sentence.text[:30]}...")
    
    try:
        # Parameters for services
        DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
        FEATURE_SERVICE_URL = os.getenv("FEATURE_SERVICE_URL", "http://localhost:8003")
        QUEUE_SERVICE_URL = os.getenv("QUEUE_SERVICE_URL", "http://localhost:8002")

        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. Store sentence in database
            try:
                db_data = {
                    "external_id": sentence.external_id,
                    "text": sentence.text
                }
                
                db_response = await client.post(
                    f"{DB_SERVICE_URL}/sentences", 
                    json=db_data
                )
                
                if db_response.status_code >= 400:
                    logger.error(f"[{request_id}] Database service error: {db_response.status_code} - {db_response.text}")
                    raise HTTPException(
                        status_code=db_response.status_code, 
                        detail=f"Database service error: {db_response.text}"
                    )
                
                db_response_data = db_response.json()
                sentence_id = db_response_data.get("id")
                
                if not sentence_id:
                    logger.error(f"[{request_id}] No sentence ID returned from database service")
                    raise HTTPException(
                        status_code=502, 
                        detail="Invalid response from database service"
                    )
                
                logger.info(f"[{request_id}] Sentence stored in database with ID: {sentence_id}")

                # 2. Extract features
                try:
                    feature_data = {
                        "sentence_id": sentence_id,
                        "text": sentence.text
                    }
                    
                    feature_response = await client.post(
                        f"{FEATURE_SERVICE_URL}/extract_features", 
                        json=feature_data
                    )
                    
                    if feature_response.status_code >= 400:
                        logger.warning(f"[{request_id}] Feature extraction service error: {feature_response.status_code} - {feature_response.text}")
                        # Continue processing even if feature extraction fails
                        response.status_code = 202
                
                except httpx.RequestError as exc:
                    logger.error(f"[{request_id}] Error connecting to feature service: {str(exc)}")
                    response.status_code = 202

                # 3. Queue for processing
                try:
                    queue_response = await client.post(
                        f"{QUEUE_SERVICE_URL}/queue", 
                        json={
                            "sentence_id": sentence_id, 
                            "external_id": sentence.external_id
                        }
                    )
                    
                    if queue_response.status_code >= 400:
                        logger.error(f"[{request_id}] Queue service error: {queue_response.status_code} - {queue_response.text}")
                        # Even if queueing fails, we still return the sentence ID
                        response.status_code = 202
                        return {
                            "id": sentence_id,
                            "status": "stored_only",
                            "message": "Sentence was stored but queueing for processing failed",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    logger.info(f"[{request_id}] Sentence queued successfully: {sentence_id}")
                    return {
                        "id": sentence_id,
                        "status": "queued_for_processing",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except httpx.RequestError as exc:
                    logger.error(f"[{request_id}] Error connecting to queue service: {str(exc)}")
                    response.status_code = 202
                    return {
                        "id": sentence_id,
                        "status": "stored_only",
                        "message": "Sentence was stored but queueing for processing failed",
                        "timestamp": datetime.now().isoformat()
                    }
                
            except httpx.RequestError as exc:
                logger.error(f"[{request_id}] Error connecting to database service: {str(exc)}")
                raise HTTPException(
                    status_code=503, 
                    detail="Database service unavailable"
                )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.get("/sentences/{sentence_id}")
async def get_sentence(sentence_id: str) -> Dict:
    """
    Retrieve a sentence by ID from the database service
    Also fetch associated features
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Retrieving sentence: {sentence_id}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. Retrieve Sentence
            start_time = time.time()
            sentence_response = await client.get(f"{DB_SERVICE_URL}/sentences/{sentence_id}")
            
            sentence_time = time.time() - start_time
            logger.info(f"[{request_id}] Sentence retrieval time: {sentence_time:.4f}s")
            
            if sentence_response.status_code == 404:
                logger.warning(f"[{request_id}] Sentence not found: {sentence_id}")
                raise HTTPException(status_code=404, detail="Sentence not found")
            elif sentence_response.status_code >= 400:
                logger.error(f"[{request_id}] Sentence retrieval error: {sentence_response.status_code}")
                raise HTTPException(
                    status_code=sentence_response.status_code, 
                    detail=f"Sentence retrieval error: {sentence_response.text}"
                )
            
            sentence_data = sentence_response.json()
            
            # 2. Retrieve Features
            try:
                start_time = time.time()
                features_response = await client.get(f"{DB_SERVICE_URL}/features/{sentence_id}")
                
                features_time = time.time() - start_time
                logger.info(f"[{request_id}] Features retrieval time: {features_time:.4f}s")
                
                if features_response.status_code == 200:
                    features_data = features_response.json()
                    sentence_data['features'] = features_data
                elif features_response.status_code != 404:
                    logger.warning(f"[{request_id}] Features retrieval error: {features_response.status_code}")
            except httpx.RequestError as exc:
                logger.error(f"[{request_id}] Error retrieving features: {str(exc)}")
            
            logger.info(f"[{request_id}] Successfully retrieved sentence {sentence_id}")
            return sentence_data
            
    except httpx.RequestError as exc:
        logger.error(f"[{request_id}] Error connecting to database service: {str(exc)}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API Gateway on port 8000")
    uvicorn.run("src.dashboard_service.app:app", host="0.0.0.0", port=8000, reload=True)