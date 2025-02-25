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
    
    async def classify_sales_funnel_relevance(self, text: str) -> dict:
        """
        Classify whether text is relevant to the sales funnel process
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are a sales funnel relevance classifier specializing in B2B and B2C customer communications.
                    
                    TASK:
                    Analyze the given text and determine if it's relevant to any stage of a sales funnel process.
                    
                    DEFINITION:
                    Sales funnel relevant text contains ANY of the following:
                    - Customer expressing interest in products/services
                    - Customer in any stage of evaluation or purchase decision-making
                    - Customer seeking product information with purchase intent
                    - References to pricing, features, comparisons, or alternatives
                    - Post-purchase feedback directly related to the buying decision
                    
                    EXAMPLES OF RELEVANT TEXT:
                    - "I'm interested in your premium plan, what features does it include?"
                    - "How does your product compare to Competitor X?"
                    - "I'm considering upgrading my subscription"
                    - "What's the pricing for enterprise customers?"
                    - "I purchased your product because of the security features"
                    
                    EXAMPLES OF NON-RELEVANT TEXT:
                    - "How do I reset my password?"
                    - "The app crashed when I tried to export data"
                    - "When will the scheduled maintenance be completed?"
                    - "Thank you for your prompt response"
                    - "Please update my email address in your system"
                    
                    OUTPUT FORMAT:
                    Return a JSON with exactly these fields:
                    - is_relevant: Boolean (true if sales funnel relevant, false if not)
                    - confidence: Confidence score between 0.0 and 1.0
                    - reasoning: Brief explanation of why the text is or isn't sales funnel relevant
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"Error in sales funnel relevance classification: {str(e)}")
            return {
                "is_relevant": False,
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    async def classify_sales_funnel_stage(self, text: str) -> dict:
        """
        Classify the sales funnel stage for a given text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are a specialized sales funnel stage classifier with expertise in customer journey analysis.
                    
                    TASK:
                    Analyze the given text and determine the most appropriate sales funnel stage.
                    
                    SALES FUNNEL STAGES DEFINITIONS:
                    1. Awareness: Customer has just discovered the product/service or is seeking initial information.
                       Keywords: discover, heard about, new to, what is, tell me about
                    
                    2. Interest: Customer shows curiosity, engagement, or is gathering more information.
                       Keywords: interested in, tell me more, features, capabilities, how does it work
                    
                    3. Consideration: Customer is actively evaluating, comparing options, or assessing fit.
                       Keywords: compare, versus, how does it stack up, pros and cons, considering
                    
                    4. Intent: Customer shows strong signals of purchase readiness or specific buying questions.
                       Keywords: pricing, cost, how to buy, discount, package options
                    
                    5. Evaluation: Customer is making final assessments, reviewing terms, or preparing to decide.
                       Keywords: terms, contract, final decision, almost ready, plan to purchase
                    
                    6. Purchase: Customer is in the buying process or has made a purchase decision.
                       Keywords: buy, purchase, checkout, sign up, subscribe, order
                    
                    EXAMPLES:
                    - "I just heard about your project management tool" = Awareness
                    - "What features does your premium plan include?" = Interest
                    - "How does your CRM compare to Salesforce?" = Consideration
                    - "What's the pricing for the enterprise package?" = Intent
                    - "I'm reviewing your contract terms before signing" = Evaluation
                    - "I'd like to purchase the annual subscription" = Purchase
                    
                    OUTPUT FORMAT:
                    Return a JSON with exactly these fields:
                    - stage: One of: "Awareness", "Interest", "Consideration", "Intent", "Evaluation", "Purchase"
                    - confidence: Confidence score between 0.0 and 1.0
                    - reasoning: Brief explanation supporting the classification
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
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
                    You are a sentiment analysis expert specializing in customer communications.
                    
                    TASK:
                    Analyze the given text and determine the sentiment with high precision.
                    
                    SENTIMENT CATEGORIES:
                    1. Positive: 
                       - Expresses satisfaction, happiness, gratitude, or enthusiasm
                       - Contains praise, appreciation, or positive feedback
                       - Uses positive emotional language or expresses good experiences
                       - Examples: "I love your product", "Thank you for the excellent service"
                    
                    2. Neutral: 
                       - Primarily factual or informational without strong emotion
                       - Balanced perspective without leaning positive or negative
                       - Routine inquiries or standard requests
                       - Examples: "What are your business hours?", "Please send me the documentation"
                    
                    3. Negative: 
                       - Expresses dissatisfaction, frustration, anger, or disappointment
                       - Contains complaints, criticism, or negative feedback
                       - Describes problems, failures, or bad experiences
                       - Examples: "Your service is terrible", "The product doesn't work as advertised"
                    
                    4. Mixed: 
                       - Contains both positive and negative elements in roughly equal measure
                       - Provides both praise and criticism in the same message
                       - Example: "I like the interface but the performance is disappointing"
                    
                    IMPORTANT CONSIDERATIONS:
                    - Focus on emotional tone, not just the content
                    - Consider intensity of emotion (mild frustration vs. anger)
                    - Polite phrasing might mask negative sentiment; look beyond formalities
                    - Context matters (e.g., "This is challenging" could be positive or negative)
                    
                    OUTPUT FORMAT:
                    Return a JSON with exactly these fields:
                    - sentiment: One of: "Positive", "Neutral", "Negative", "Mixed"
                    - confidence: Confidence score between 0.0 and 1.0
                    - reasoning: Brief explanation of the sentiment identification
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
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
                    You are an intent classification specialist focusing on customer communications.
                    
                    TASK:
                    Analyze the given text and determine the primary customer intent.
                    
                    INTENT CATEGORIES:
                    1. Support: 
                       - Customer needs help solving a problem
                       - Technical assistance requests
                       - Usage questions or guidance
                       - Examples: "How do I export my data?", "The app keeps crashing"
                    
                    2. Purchase: 
                       - Inquiry directly related to buying products/services
                       - Questions about pricing, plans, or payment
                       - Intent to subscribe, renew, or upgrade
                       - Examples: "How much is the premium plan?", "I want to upgrade my account"
                    
                    3. Information: 
                       - Seeking factual details or clarification
                       - General questions not tied to immediate problems
                       - Knowledge gathering without clear purchase intent
                       - Examples: "What are your business hours?", "Do you have an API?"
                    
                    4. Complaint: 
                       - Expressing dissatisfaction about product/service
                       - Reporting negative experiences
                       - Seeking resolution for a negative situation
                       - Examples: "Your service is unreliable", "I'm unhappy with the quality"
                    
                    5. Feedback: 
                       - Providing suggestions, opinions, or reviews
                       - Offering constructive input 
                       - Sharing experiences without requesting specific action
                       - Examples: "I think you should add this feature", "Your product is great"
                    
                    6. General: 
                       - Conversational remarks not fitting other categories
                       - Greetings, acknowledgments, or pleasantries
                       - Ambiguous messages without clear purpose
                       - Examples: "Thanks for your time", "Looking forward to hearing from you"
                    
                    IMPORTANT GUIDELINES:
                    - Identify the dominant intent if multiple are present
                    - Consider the primary action the customer wants
                    - Look for specific request patterns and keywords
                    - Context matters more than individual words
                    
                    OUTPUT FORMAT:
                    Return a JSON with exactly these fields:
                    - intent: One of: "Support", "Purchase", "Information", "Complaint", "Feedback", "General"
                    - confidence: Confidence score between 0.0 and 1.0 
                    - reasoning: Brief explanation of the intent identification
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
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
                    You are a business impact analyst specializing in customer communications assessment.
                    
                    TASK:
                    Analyze the given text and determine its potential business impact by assessing:
                    - Revenue implications
                    - Customer retention risk/opportunity
                    - Operational impact
                    - Strategic importance
                    - Urgency/time sensitivity
                    
                    BUSINESS IMPACT LEVELS:
                    1. High: 
                       - Directly affects revenue (large purchases, renewals, cancelations)
                       - Involves high-value customers or large accounts
                       - Could significantly influence brand reputation
                       - Indicates strong potential for upsell/cross-sell
                       - Examples: "We're considering implementing your solution company-wide", 
                                  "I'm evaluating whether to renew our enterprise contract"
                    
                    2. Medium: 
                       - Moderate revenue implications
                       - Affects customer satisfaction but not immediate churn risk
                       - Opportunity for relationship strengthening
                       - Operational improvements possible
                       - Examples: "I'd like to add 3 more licenses", 
                                  "This feature would make your product much more useful to us"
                    
                    3. Low: 
                       - Minimal direct business consequences
                       - Routine inquiries or minor issues
                       - Limited short-term revenue impact
                       - Examples: "How do I change my password?", 
                                  "What are your holiday hours?"
                    
                    4. Critical: 
                       - Severe churn risk with immediate attention required
                       - Legal or compliance implications
                       - Major system/service failures affecting multiple customers
                       - Potential for significant negative publicity
                       - Examples: "We're experiencing a complete system outage affecting our production", 
                                  "I'm prepared to cancel our contract due to ongoing issues"
                    
                    5. Neutral: 
                       - Informational exchanges without clear business impact
                       - General comments or acknowledgments
                       - Examples: "Thanks for the information", 
                                  "I received your newsletter"
                    
                    IMPORTANT CONSIDERATIONS:
                    - Consider customer account size/value when mentioned
                    - Assess urgency signals in the language
                    - Evaluate potential long-term impact beyond immediate request
                    - Consider competitive implications if mentioned
                    
                    OUTPUT FORMAT:
                    Return a JSON with exactly these fields:
                    - impact: One of: "High", "Medium", "Low", "Critical", "Neutral"
                    - confidence: Confidence score between 0.0 and 1.0
                    - reasoning: Brief explanation of the impact assessment
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
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
            
            # STAGE 1: Check relevance to sales funnel
            relevance_start = time.time()
            relevance_result = await sentence_label_classifier.classify_sales_funnel_relevance(text)
            is_relevant = relevance_result.get("is_relevant", False)
            relevance_confidence = relevance_result.get("confidence", 0.0)
            
            relevance_time = time.time() - relevance_start
            logger.info(f"[RELEVANCE] Determined sales funnel relevance in {relevance_time:.3f}s: {is_relevant}")
            
            # Initialize label data with relevance information and NULL values for funnel stage
            label_data = {
                "sentence_id": sentence_id,
                "is_sales_funnel_relevant": is_relevant,
                "is_sales_funnel_relevant_confidence": relevance_confidence,
                # Important: Set sales funnel fields to None for non-relevant sentences
                "sales_funnel_stage": None,
                "sales_funnel_confidence": None,
                # Initialize other fields (will be updated regardless of relevance)
                "sentiment": "",
                "sentiment_confidence": 0.0,
                "intent": "",
                "intent_confidence": 0.0,
                "business_impact": "",
                "business_impact_confidence": 0.0
            }
            
            # Run standard classifications for ALL sentences (regardless of relevance)
            classification_start = time.time()
            
            # Always run these classifications concurrently
            sentiment_task = asyncio.create_task(sentence_label_classifier.classify_sentiment(text))
            intent_task = asyncio.create_task(sentence_label_classifier.classify_intent(text))
            business_impact_task = asyncio.create_task(sentence_label_classifier.classify_business_impact(text))
            
            # Only run sales funnel stage classification if the sentence is relevant
            sales_funnel_task = None
            if is_relevant:
                logger.info(f"[CLASSIFY] Sentence {sentence_id} is sales funnel relevant, including funnel stage classification")
                sales_funnel_task = asyncio.create_task(sentence_label_classifier.classify_sales_funnel_stage(text))
            
            # Wait for the standard classifications (these run for all sentences)
            sentiment = await sentiment_task
            intent = await intent_task
            business_impact = await business_impact_task
            
            # Update label data with the standard classifications
            label_data.update({
                "sentiment": str(sentiment.get("sentiment", "")),
                "sentiment_confidence": float(sentiment.get("confidence", 0.0)),
                "intent": str(intent.get("intent", "")),
                "intent_confidence": float(intent.get("confidence", 0.0)),
                "business_impact": str(business_impact.get("impact", "")),
                "business_impact_confidence": float(business_impact.get("confidence", 0.0))
            })
            
            # If relevant, wait for and include the sales funnel stage
            if is_relevant and sales_funnel_task:
                sales_funnel = await sales_funnel_task
                label_data.update({
                    "sales_funnel_stage": str(sales_funnel.get("stage", "")),
                    "sales_funnel_confidence": float(sales_funnel.get("confidence", 0.0))
                })
                logger.info(f"[CLASSIFY] Included sales funnel stage: {label_data['sales_funnel_stage']}")
            
            classification_time = time.time() - classification_start
            logger.info(f"[CLASSIFY] Completed classification in {classification_time:.3f}s")
            
            # Store all results in the database
            update_start = time.time()
            logger.debug(f"[UPDATE] Sending label results to database for sentence {sentence_id}")
            logger.debug(f"[UPDATE] Label data: {json.dumps(label_data)}")
            
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
            
            # Publish to processed queue
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
                    "is_sales_funnel_relevant": label_data["is_sales_funnel_relevant"],
                    "sales_funnel_stage": label_data["sales_funnel_stage"],
                    "business_impact": label_data["business_impact"],
                    "intent": label_data["intent"],
                    "sentiment": label_data["sentiment"]
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