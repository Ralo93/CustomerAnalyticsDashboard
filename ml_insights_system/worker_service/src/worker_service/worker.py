import os
import json
import asyncio
import logging
import time
import uuid
import httpx
from datetime import datetime
from dotenv import load_dotenv
import aio_pika
from priority_classifier import PriorityClassifier

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(thread)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)
logger = logging.getLogger(__name__)

# Add file handler
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler(f'logs/sentence_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(thread)d] - %(message)s'))
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

# Configuration
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("QUEUE_NAME", "sentence_processing")
PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed")
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")

# Initialize the priority classifier
priority_classifier = PriorityClassifier()

async def process_sentence(sentence_id: str, connection=None):
    """
    Process a sentence - includes ML processing and priority classification
    """
    start_time = time.time()
    logger.info(f"[START] Processing sentence: {sentence_id}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get the sentence
            fetch_start = time.time()
            logger.debug(f"[FETCH] Retrieving sentence {sentence_id} from database")
            response = await client.get(f"{DB_SERVICE_URL}/sentences/{sentence_id}")
            
            if response.status_code != 200:
                logger.error(f"[ERROR] Failed to fetch sentence {sentence_id}: {response.status_code} - {response.text}")
                return False
            
            sentence_data = response.json()
            text = sentence_data.get("text", "")
            
            fetch_time = time.time() - fetch_start
            logger.info(f"[FETCH] Retrieved sentence: {sentence_id} in {fetch_time:.3f}s - Text preview: {text[:30]}...")
            
            # Perform basic processing
            # Simulate some processing time
            processing_start = time.time()
            logger.debug(f"[PROCESS] Starting basic processing for sentence {sentence_id}")
            await asyncio.sleep(1)
            
            # Example result - replace with actual ML processing result
            processing_result = {
                "sentiment": "positive" if "good" in text.lower() else "negative" if "bad" in text.lower() else "neutral",
                "word_count": len(text.split()),
                "processing_status": "completed"
            }
            processing_time = time.time() - processing_start
            logger.info(f"[PROCESS] Completed basic processing in {processing_time:.3f}s: {sentence_id}")
            
            # Classify the priority using OpenAI
            priority_start = time.time()
            logger.debug(f"[PRIORITY] Starting priority classification for sentence {sentence_id}")
            priority_result = await priority_classifier.classify_priority(text)
            priority_time = time.time() - priority_start
            logger.info(f"[PRIORITY] Priority classification completed in {priority_time:.3f}s for {sentence_id}: {priority_result.get('priority', 'unknown')}")
            
            # Combine the results
            combined_result = {
                **processing_result,
                "priority": priority_result.get("priority", "normal"),
                "intent_type": priority_result.get("intent_type", "none"),
                "priority_reasoning": priority_result.get("reasoning", "")
            }
            
            # Update the sentence with processing results
            update_start = time.time()
            logger.debug(f"[UPDATE] Sending results to database for sentence {sentence_id}")
            update_response = await client.patch(
                f"{DB_SERVICE_URL}/sentences/{sentence_id}",
                json=combined_result
            )
            
            if update_response.status_code not in (200, 201, 204):
                logger.error(f"[ERROR] Failed to update sentence {sentence_id}: {update_response.status_code} - {update_response.text}")
                return False
            
            update_time = time.time() - update_start
            total_time = time.time() - start_time
            logger.info(f"[UPDATE] Database updated in {update_time:.3f}s for sentence {sentence_id}")
            logger.info(f"[COMPLETE] Total processing for sentence {sentence_id} took {total_time:.3f}s, priority: {combined_result['priority']}")


            # Publish to processed queue
            try:
                # Use the connection passed in, or get the global one if available
                if connection is None:
                    # This assumes global_connection is available in this scope
                    # If not, you'll need to modify this part to get the connection
                    if 'global_connection' in globals():
                        connection = global_connection
                    else:
                        logger.warning(f"[WARN] No RabbitMQ connection available to publish processed notification")
                        return True
                
                # Define the processed queue name
                PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed")
                
                # Create a channel
                channel = await connection.channel()
                
                # Declare the queue
                await channel.declare_queue(
                    PROCESSED_QUEUE_NAME,
                    durable=True  # Queue survives broker restart
                )
                
                # Create the message with processed sentence info
                message_body = json.dumps({
                    "sentence_id": sentence_id,
                    "processed_at": datetime.now().isoformat(),
                    "priority": combined_result["priority"],
                    "intent_type": combined_result["intent_type"],
                    "sentiment": combined_result["sentiment"]
                })
                
                # Publish to the queue
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message_body.encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        message_id=str(uuid.uuid4())
                    ),
                    routing_key=PROCESSED_QUEUE_NAME
                )
                
                logger.info(f"[RABBITMQ] Published processed notification for sentence {sentence_id} to '{PROCESSED_QUEUE_NAME}' queue")
            except Exception as e:
                logger.error(f"[ERROR] Failed to publish processed notification: {str(e)}", exc_info=True)
                # Continue processing, don't fail just because notification failed
            
            return True
            
    except httpx.RequestError as e:
        total_time = time.time() - start_time
        logger.error(f"[ERROR] Network error while processing sentence {sentence_id} after {total_time:.3f}s: {str(e)}")
        return False
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[ERROR] Error processing sentence {sentence_id} after {total_time:.3f}s: {str(e)}", exc_info=True)
        return False

async def on_message(message: aio_pika.IncomingMessage):
    """
    Process incoming messages from RabbitMQ
    """
    message_id = message.message_id or "unknown"
    message_receive_time = datetime.now().isoformat()
    logger.info(f"[RABBITMQ] Received message {message_id} at {message_receive_time}")
    logger.debug(f"[RABBITMQ] Message {message_id} raw body: {message.body}")

    async with message.process():
        start_time = time.time()
        try:
            # Decode the message body
            body = message.body.decode()
            data = json.loads(body)
            
            sentence_id = data.get("sentence_id")
            if not sentence_id:
                logger.error(f"[ERROR] Received message {message_id} without sentence_id: {body}")
                return
            
            logger.info(f"[RABBITMQ] Processing message {message_id} for sentence: {sentence_id}")

            connection = message.channel.connection
            
            # Process the sentence
            success = await process_sentence(sentence_id)
            
            processing_time = time.time() - start_time
            if not success:
                # Normally, we'd reject and requeue, but for simplicity we'll 
                # just log the error and consider it processed
                logger.warning(f"[WARN] Failed to process sentence {sentence_id} after {processing_time:.3f}s")
            else:
                logger.info(f"[SUCCESS] Successfully processed message {message_id} for sentence {sentence_id} in {processing_time:.3f}s")
            
        except json.JSONDecodeError:
            logger.error(f"[ERROR] Failed to decode message {message_id}: {message.body}")
        except Exception as e:
            logger.error(f"[ERROR] Error handling message {message_id}: {str(e)}", exc_info=True)

async def main():
    """
    Main function to run the worker
    """

    global global_connection


    start_time = datetime.now().isoformat()
    logger.info(f"[STARTUP] Starting sentence processing worker at {start_time}")
    logger.info(f"[CONFIG] RABBITMQ_URL: {RABBITMQ_URL}")
    logger.info(f"[CONFIG] QUEUE_NAME: {QUEUE_NAME}")
    logger.info(f"[CONFIG] DB_SERVICE_URL: {DB_SERVICE_URL}")
    
    try:
        # Connect to RabbitMQ
        connection_start = time.time()
        logger.info("[RABBITMQ] Connecting to RabbitMQ...")
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        connection_time = time.time() - connection_start
        logger.info(f"[RABBITMQ] Connected to RabbitMQ in {connection_time:.3f}s")

        global_connection = connection
        
        # Create channel
        channel = await connection.channel()
        
        # Set QoS (prefetch_count)
        await channel.set_qos(prefetch_count=1)
        
        # Declare the queue
        queue = await channel.declare_queue(
            QUEUE_NAME,
            durable=True  # Queue survives broker restart
        )
        
        logger.info(f"[RABBITMQ] Listening on queue: {QUEUE_NAME}")
        logger.info(f"[RABBITMQ] Publishing processed notifications to: {PROCESSED_QUEUE_NAME}")
        
        # Start consuming messages
        await queue.consume(on_message)
        
        logger.info("[READY] Worker is now ready to process messages")
        
        try:
            # Keep the worker running
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("[SHUTDOWN] Worker received shutdown signal")
        finally:
            # Close the connection when the worker is stopped
            logger.info("[SHUTDOWN] Closing RabbitMQ connection...")
            await connection.close()
            logger.info("[SHUTDOWN] RabbitMQ connection closed")
    except Exception as e:
        logger.critical(f"[CRITICAL] Failed to start worker: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] Worker stopped by keyboard interrupt")
    except Exception as e:
        logger.critical(f"[CRITICAL] Unhandled exception in main: {str(e)}", exc_info=True)