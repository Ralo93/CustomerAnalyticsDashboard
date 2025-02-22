import os
import json
import time
import pika
import logging
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8005"))

# Create FastAPI app
app = FastAPI(title="Simple RabbitMQ Bridge Service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Message storage
class BridgeState:
    def __init__(self):
        self.recent_messages: List[Dict] = []
        self.max_messages = 50
        self.last_update = time.time()
        self.connected_to_rabbitmq = False
        self.total_messages_received = 0

# Create global state
state = BridgeState()

# Message callback function
def process_message(ch, method, properties, body):
    try:
        # Parse message
        data = json.loads(body)
        
        # Add timestamp for when bridge received the message
        data['bridge_received_at'] = datetime.now().isoformat()
        
        # Store the message
        state.recent_messages.append(data)
        if len(state.recent_messages) > state.max_messages:
            state.recent_messages.pop(0)  # Remove oldest
            
        state.last_update = time.time()
        state.total_messages_received += 1
        
        logger.info(f"Received message for sentence: {data.get('sentence_id')}")
        
        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        # Acknowledge anyway to avoid message queue buildup
        ch.basic_ack(delivery_tag=method.delivery_tag)

# RabbitMQ listener thread function
def rabbitmq_listener():
    while True:
        try:
            # Connect to RabbitMQ
            logger.info(f"Connecting to RabbitMQ at {RABBITMQ_URL}")
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            
            # Declare the queue
            channel.queue_declare(queue=PROCESSED_QUEUE_NAME, durable=True)
            
            # Set up consumer
            channel.basic_consume(
                queue=PROCESSED_QUEUE_NAME,
                on_message_callback=process_message
            )
            
            # Update state
            state.connected_to_rabbitmq = True
            logger.info(f"Connected to RabbitMQ, listening on queue: {PROCESSED_QUEUE_NAME}")
            
            # Start consuming
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
                break
            except Exception as e:
                logger.error(f"Error consuming messages: {str(e)}")
                try:
                    connection.close()
                except:
                    pass
        except Exception as e:
            state.connected_to_rabbitmq = False
            logger.error(f"Error connecting to RabbitMQ: {str(e)}")
            
        # Wait before reconnecting
        time.sleep(5)

# Start RabbitMQ listener in background thread
def start_rabbitmq_listener():
    thread = threading.Thread(target=rabbitmq_listener, daemon=True)
    thread.start()
    logger.info("Started RabbitMQ listener thread")

# API Models
class Message(BaseModel):
    sentence_id: str
    processed_at: str
    priority: Optional[str] = None
    intent_type: Optional[str] = None
    sentiment: Optional[str] = None
    bridge_received_at: Optional[str] = None

class ServiceStatus(BaseModel):
    status: str
    connected_to_rabbitmq: bool
    total_messages_received: int
    last_update: str
    queue_name: str

# Endpoint to get recent messages
@app.get("/messages", response_model=List[Message])
async def get_messages():
    return state.recent_messages

# Health check endpoint
@app.get("/health", response_model=ServiceStatus)
async def health_check():
    return {
        "status": "healthy" if state.connected_to_rabbitmq else "degraded",
        "connected_to_rabbitmq": state.connected_to_rabbitmq,
        "total_messages_received": state.total_messages_received,
        "last_update": datetime.fromtimestamp(state.last_update).isoformat(),
        "queue_name": PROCESSED_QUEUE_NAME
    }

# App startup event
@app.on_event("startup")
def startup_event():
    logger.info("Starting bridge service")
    start_rabbitmq_listener()

# Run the service
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting bridge service on port {BRIDGE_PORT}")
    uvicorn.run("bridge_service:app", host="0.0.0.0", port=BRIDGE_PORT, reload=True)

    #src.dashboard_service.app:app