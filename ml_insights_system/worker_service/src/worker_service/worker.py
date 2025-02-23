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
import openai

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to see more information
    format='%(asctime)s - %(levelname)s - %(message)s'
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


class SentenceLabelClassifier:
    """
    A class to classify various dimensions of a sentence
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info("SentenceLabelClassifier initialized with OpenAI API")
    
    async def classify_sales_funnel_stage(self, text: str) -> dict:
        """
        Classify the sales funnel stage for a given text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are a sales funnel stage classifier. Analyze the given text and determine 
                    the most appropriate sales funnel stage. The stages are:
                    
                    1. Awareness: Initial discovery of the product/service
                    2. Interest: Showing curiosity or initial engagement
                    3. Consideration: Actively evaluating the offering
                    4. Intent: Strong indication of potential purchase
                    5. Evaluation: Comparing with alternatives
                    6. Purchase: Ready to buy or in purchase process
                    
                    Return a JSON with:
                    - stage: The identified sales funnel stage
                    - confidence: Confidence score (0-1)
                    - reasoning: Brief explanation of the classification
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"Error in sales funnel stage classification: {str(e)}")
            return {
                "stage": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    async def classify_sentiment(self, text: str) -> dict:
        """
        Classify the sentiment of the given text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are a sentiment classifier. Analyze the given text and determine 
                    the sentiment with high precision. The possible sentiments are:
                    
                    1. Positive: Expresses satisfaction, excitement, or enthusiasm
                    2. Neutral: Factual or balanced without strong emotions
                    3. Negative: Expresses dissatisfaction, frustration, or anger
                    4. Mixed: Contains conflicting emotional tones
                    
                    Return a JSON with:
                    - sentiment: The identified sentiment
                    - confidence: Confidence score (0-1)
                    - reasoning: Brief explanation of the sentiment
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"Error in sentiment classification: {str(e)}")
            return {
                "sentiment": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    async def classify_intent(self, text: str) -> dict:
        """
        Classify the intent of the given text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are an intent classifier. Analyze the given text and determine 
                    the primary customer intent. The possible intents are:
                    
                    1. Support: Seeking help or customer service
                    2. Purchase: Interested in buying or learning about products
                    3. Information: Requesting details or clarification
                    4. Complaint: Expressing dissatisfaction or reporting an issue
                    5. Feedback: Providing constructive input or suggestions
                    6. General: Conversational or not clearly categorized
                    
                    Return a JSON with:
                    - intent: The identified intent
                    - confidence: Confidence score (0-1)
                    - reasoning: Brief explanation of the intent
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    async def classify_business_impact(self, text: str) -> dict:
        """
        Classify the potential business impact of the given text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are a business impact classifier. Analyze the given text and determine 
                    the potential business impact. The possible impact levels are:
                    
                    1. High: Significant potential for revenue, retention, or strategic change
                    2. Medium: Moderate potential impact on business operations
                    3. Low: Minimal immediate business implications
                    4. Critical: Urgent issues requiring immediate attention
                    5. Neutral: No clear significant impact
                    
                    Return a JSON with:
                    - impact: The identified business impact level wording, e.g. High, Medium etc.
                    - confidence: Confidence score (0-1)
                    - reasoning: Brief explanation of the impact assessment
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"Error in business impact classification: {str(e)}")
            return {
                "impact": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }

# Initialize the sentence label classifier
sentence_label_classifier = SentenceLabelClassifier()

async def process_sentence(sentence_id: str, connection=None):
    """
    Process a sentence - includes ML processing and comprehensive labeling
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
            
            # Classify various dimensions
            classification_start = time.time()
            
            # Run classifications concurrently
            sales_funnel_task = asyncio.create_task(sentence_label_classifier.classify_sales_funnel_stage(text))
            sentiment_task = asyncio.create_task(sentence_label_classifier.classify_sentiment(text))
            intent_task = asyncio.create_task(sentence_label_classifier.classify_intent(text))
            business_impact_task = asyncio.create_task(sentence_label_classifier.classify_business_impact(text))
            
            # Wait for all classifications
            sales_funnel = await sales_funnel_task
            sentiment = await sentiment_task
            intent = await intent_task
            business_impact = await business_impact_task

            classification_time = time.time() - classification_start
            logger.info(f"[CLASSIFY] Completed classifications in {classification_time:.3f}s")
            
            label_data = {
                "sentence_id": sentence_id,
                "sales_funnel_stage": str(sales_funnel.get("stage", "")),
                "sales_funnel_confidence": float(sales_funnel.get("confidence", 0.0)),
                "sentiment": str(sentiment.get("sentiment", "")),
                "sentiment_confidence": float(sentiment.get("confidence", 0.0)),
                "intent": str(intent.get("intent", "")),
                "intent_confidence": float(intent.get("confidence", 0.0)),
                "business_impact": str(business_impact.get("impact", "")),
                "business_impact_confidence": float(business_impact.get("confidence", 0.0))
            }
            
            # Update the sentence label in the database
            update_start = time.time()
            logger.debug(f"[UPDATE] Sending label results to database for sentence {sentence_id}")
            update_response = await client.post(
                f"{DB_SERVICE_URL}/labels",
                json=label_data
            )
            
            if update_response.status_code not in (200, 201, 204):
                logger.error(f"[ERROR] Failed to create sentence label for {sentence_id}: {update_response.status_code} - {update_response.text}")
                return False
            
            update_time = time.time() - update_start
            total_time = time.time() - start_time
            logger.info(f"[UPDATE] Sentence label created in {update_time:.3f}s for sentence {sentence_id}")
            logger.info(f"[COMPLETE] Total processing for sentence {sentence_id} took {total_time:.3f}s")

            # Publish to processed queue (similar to previous implementation)
            try:
                if connection is None:
                    if 'global_connection' in globals():
                        connection = global_connection
                    else:
                        logger.warning(f"[WARN] No RabbitMQ connection available to publish processed notification")
                        return True
                
                PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed")
                
                # Create a channel
                channel = await connection.channel()
                
                # Declare the queue
                await channel.declare_queue(
                    PROCESSED_QUEUE_NAME,
                    durable=True
                )
                
                # Create the message with processed sentence info
                message_body = json.dumps({
                    "sentence_id": sentence_id,
                    "processed_at": datetime.now().isoformat(),
                    "business_impact": label_data["business_impact"],
                    "intent": label_data["intent"],
                    "sentiment": label_data["sentiment"],
                    "sales_funnel_stage": label_data["sales_funnel_stage"]
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
            
            return True
            
    except httpx.RequestError as e:
        total_time = time.time() - start_time
        logger.error(f"[ERROR] Network error while processing sentence {sentence_id} after {total_time:.3f}s: {str(e)}")
        return False
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[ERROR] Error processing sentence {sentence_id} after {total_time:.3f}s: {str(e)}", exc_info=True)
        return False

# Rest of the code remains the same as in the original worker.py
# (include the on_message, main, and other functions from the original file)

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
            
            # Extract sentence_id, using a more robust approach
            sentence_id = data.get('sentence_id') or data.get('id')
            
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