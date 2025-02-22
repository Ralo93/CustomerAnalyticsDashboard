import os
import json
import logging
import time
import uuid
from typing import Dict, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
import aio_pika
from pydantic import BaseModel
from dotenv import load_dotenv

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# RabbitMQ configuration
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("QUEUE_NAME", "sentence_processing")

app = FastAPI(title="Queue Service")

# Connection to RabbitMQ
rabbitmq_connection = None
rabbitmq_channel = None

class TaskPayload(BaseModel):
    """Model for task data"""
    sentence_id: str
    priority: Optional[str] = "normal"


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
    """Initialize RabbitMQ connection on startup"""
    global rabbitmq_connection, rabbitmq_channel
    
    logger.info("Queue service starting up...")
    logger.info(f"RabbitMQ URL: {RABBITMQ_URL}")
    logger.info(f"Queue Name: {QUEUE_NAME}")
    
    try:
        # Connect to RabbitMQ
        start_time = time.time()
        logger.info(f"Connecting to RabbitMQ at {RABBITMQ_URL}")
        rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
        
        # Create channel
        rabbitmq_channel = await rabbitmq_connection.channel()
        
        # Declare queue
        await rabbitmq_channel.declare_queue(
            QUEUE_NAME, 
            durable=True  # Queue survives broker restart
        )
        
        connection_time = time.time() - start_time
        logger.info(f"Successfully connected to RabbitMQ in {connection_time:.4f}s and declared queue: {QUEUE_NAME}")
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
        # We don't want to crash the app if RabbitMQ is down on startup
        # The service will attempt to reconnect when a publish is attempted


@app.on_event("shutdown")
async def shutdown_event():
    """Close RabbitMQ connection on shutdown"""
    global rabbitmq_connection
    
    logger.info("Queue service shutting down...")
    if rabbitmq_connection:
        await rabbitmq_connection.close()
        logger.info("RabbitMQ connection closed")


async def reconnect_rabbitmq():
    """Reconnect to RabbitMQ if connection is lost"""
    global rabbitmq_connection, rabbitmq_channel
    
    logger.info("Attempting to reconnect to RabbitMQ...")
    try:
        start_time = time.time()
        
        if rabbitmq_connection and not rabbitmq_connection.is_closed:
            await rabbitmq_connection.close()
            
        rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
        rabbitmq_channel = await rabbitmq_connection.channel()
        
        # Ensure queue exists
        await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)
        
        connection_time = time.time() - start_time
        logger.info(f"Successfully reconnected to RabbitMQ in {connection_time:.4f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to reconnect to RabbitMQ: {str(e)}")
        return False


async def publish_to_queue(task: TaskPayload):
    """Publish a message to the RabbitMQ queue"""
    global rabbitmq_connection, rabbitmq_channel
    
    operation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{operation_id}] Publishing task {task.sentence_id} to queue")
    
    # Check if we're connected
    if not rabbitmq_connection or rabbitmq_connection.is_closed:
        logger.warning(f"[{operation_id}] RabbitMQ connection is closed or not initialized, attempting to reconnect")
        success = await reconnect_rabbitmq()
        if not success:
            logger.error(f"[{operation_id}] Failed to reconnect to RabbitMQ")
            raise HTTPException(status_code=503, detail="Queue service unavailable")
    
    try:
        # Create a message
        message_body = json.dumps(task.dict()).encode()
        message = aio_pika.Message(
            body=message_body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # Message survives broker restart
            content_type="application/json",
            headers={"published_at": datetime.utcnow().isoformat()}
        )
        
        start_time = time.time()
        # Publish to the queue
        await rabbitmq_channel.default_exchange.publish(
            message, 
            routing_key=QUEUE_NAME
        )
        
        publish_time = time.time() - start_time
        logger.info(f"[{operation_id}] Message published to queue in {publish_time:.4f}s: {task.sentence_id}")
        return True
    except Exception as e:
        logger.error(f"[{operation_id}] Failed to publish message: {str(e)}")
        # Try to reconnect
        logger.info(f"[{operation_id}] Attempting to reconnect before giving up")
        await reconnect_rabbitmq()
        raise HTTPException(status_code=503, detail="Failed to queue task")


@app.post("/queue")
async def queue_task(task_data: TaskPayload, background_tasks: BackgroundTasks) -> Dict:
    """Queue a task for processing"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request to queue task: {task_data.sentence_id}")
    
    try:
        start_time = time.time()
        # Publish to RabbitMQ
        await publish_to_queue(task_data)
        
        process_time = time.time() - start_time
        logger.info(f"[{request_id}] Task successfully queued in {process_time:.4f}s: {task_data.sentence_id}")
        
        return {
            "status": "queued",
            "task_id": task_data.sentence_id,
            "queue": QUEUE_NAME,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error queueing task: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    global rabbitmq_connection
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Health check requested")
    
    # Check RabbitMQ connection
    if not rabbitmq_connection or rabbitmq_connection.is_closed:
        logger.warning(f"[{request_id}] RabbitMQ connection is closed or not initialized during health check")
        try:
            start_time = time.time()
            success = await reconnect_rabbitmq()
            check_time = time.time() - start_time
            
            if not success:
                logger.error(f"[{request_id}] Health check failed: RabbitMQ connection failed after {check_time:.4f}s")
                return {
                    "status": "unhealthy", 
                    "detail": "RabbitMQ connection failed",
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"[{request_id}] Health check failed: {str(e)}")
            return {
                "status": "unhealthy", 
                "detail": "RabbitMQ connection failed",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    logger.info(f"[{request_id}] Health check passed")
    return {
        "status": "healthy",
        "service": "queue_service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/queue/status")
async def queue_status() -> Dict:
    """Get queue status"""
    global rabbitmq_channel
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Queue status requested")
    
    if not rabbitmq_channel or rabbitmq_channel.is_closed:
        logger.warning(f"[{request_id}] RabbitMQ channel is closed or not initialized during status check")
        try:
            start_time = time.time()
            success = await reconnect_rabbitmq()
            reconnect_time = time.time() - start_time
            
            if not success:
                logger.error(f"[{request_id}] Queue status check failed: reconnection failed after {reconnect_time:.4f}s")
                raise HTTPException(status_code=503, detail="Queue service unavailable")
                
            logger.info(f"[{request_id}] Successfully reconnected to RabbitMQ in {reconnect_time:.4f}s")
        except Exception as e:
            logger.error(f"[{request_id}] Queue status check failed: {str(e)}")
            raise HTTPException(status_code=503, detail="Queue service unavailable")
    
    try:
        start_time = time.time()
        # Get queue info
        queue = await rabbitmq_channel.declare_queue(QUEUE_NAME, passive=True)
        query_time = time.time() - start_time
        
        message_count = queue.declaration_result.message_count
        consumer_count = queue.declaration_result.consumer_count
        
        logger.info(f"[{request_id}] Queue status retrieved in {query_time:.4f}s: {message_count} messages, {consumer_count} consumers")
        
        return {
            "queue_name": QUEUE_NAME,
            "message_count": message_count,
            "consumer_count": consumer_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"[{request_id}] Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get queue status")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Queue Service on port 8002")
    uvicorn.run("service:app", host="0.0.0.0", port=8002, reload=True)