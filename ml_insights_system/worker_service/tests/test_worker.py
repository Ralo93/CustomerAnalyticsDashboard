import unittest
import requests
import uuid
import logging
import json
import aio_pika
import asyncio
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service URLs
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
DB_SERVICE_URL = "http://localhost:8001"
WORKER_QUEUE_NAME = "sentence_processing"

class WorkerServiceTests(unittest.TestCase):
    
    def create_sentence(self, text):
        """Helper method to create a sentence in the database service"""
        sentence_data = {
            "external_id": str(uuid.uuid4()),  # Generate unique external_id
            "text": text
        }
        response = requests.post(f"{DB_SERVICE_URL}/sentences", json=sentence_data)
        self.assertEqual(response.status_code, 201)
        return response.json()

    def setUp(self):
        """Setup before each test - verify services are running"""
        try:
            # Check database service
            db_response = requests.get(f"{DB_SERVICE_URL}/health")
            if db_response.status_code != 200:
                self.fail("Database service is not running.")
            
            # Check RabbitMQ connection
            try:
                connection = self.sync_connect_rabbitmq()
                connection.close()
            except Exception as e:
                self.fail(f"RabbitMQ connection failed: {e}")
            
            logger.info("Services are running. Starting tests.")
            
        except requests.ConnectionError:
            self.fail("Could not connect to services. Please start all services before running tests.")

    def sync_connect_rabbitmq(self):
        """Synchronous RabbitMQ connection for health check"""
        import pika
        connection_params = pika.ConnectionParameters('localhost')
        return pika.BlockingConnection(connection_params)

    async def send_sentence_to_queue(self, sentence_data):
        """Send a sentence to the processing queue"""
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        async with connection:
            channel = await connection.channel()
            
            test_message = {
                "id": sentence_data['id'],  # Use database-generated ID
                "external_id": sentence_data['external_id'],
                "text": sentence_data['text']
            }
            
            message = aio_pika.Message(
                body=json.dumps(test_message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )
            
            await channel.default_exchange.publish(
                message,
                routing_key=WORKER_QUEUE_NAME
            )
            
            logger.info(f"Sent sentence {sentence_data['id']} to processing queue")

    def test_worker_sentence_processing(self):
        """Test full sentence processing workflow"""
        test_texts = [
            "I'm interested in purchasing your product",
            "The service was not meeting my expectations",
            "Can you provide more information about your pricing?"
        ]
        
        for text in test_texts:
            # Create sentence
            sentence_data = self.create_sentence(text)
            
            # Run async method to send to queue
            asyncio.run(self.send_sentence_to_queue(sentence_data))
            
            # Wait for processing
            asyncio.run(self.wait_and_verify_label(sentence_data['id']))

    async def wait_and_verify_label(self, sentence_id, timeout=30):
        """
        Wait for and verify label creation
        
        Args:
            sentence_id (str): ID of the sentence to check
            timeout (int): Maximum wait time in seconds
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    # Retrieve the label
                    label_response = await client.get(f"{DB_SERVICE_URL}/labels/{sentence_id}")
                    
                    # If label doesn't exist, wait and continue
                    if label_response.status_code != 200:
                        await asyncio.sleep(1)
                        continue
                    
                    label_data = label_response.json()
                    
                    # Verify the sentence_id matches
                    self.assertEqual(
                        str(label_data['sentence_id']), 
                        str(sentence_id), 
                        "Sentence ID mismatch in label"
                    )
                    
                    # Verify label has been processed
                    self.assertIsNotNone(label_data['sales_funnel_stage'], "Sales funnel stage not processed")
                    self.assertIsNotNone(label_data['sentiment'], "Sentiment not processed")
                    self.assertIsNotNone(label_data['intent'], "Intent not processed")
                    self.assertIsNotNone(label_data['business_impact'], "Business impact not processed")
                    
                    logger.info(f"Label successfully created for sentence {sentence_id}")
                    return
            
            except Exception as e:
                logger.error(f"Error checking label: {e}")
                await asyncio.sleep(1)
        
        # If we exit the loop, it means we timed out
        self.fail(f"Timed out waiting for label creation for sentence {sentence_id}")

    def test_multiple_sentence_processing(self):
        """Test processing multiple sentences concurrently"""
        sentences = [
            "This is a great product with amazing features",
            "I'm experiencing issues with the current service",
            "What are the pricing options available?",
            "Can you help me understand the technical specifications?"
        ]
        
        # Create sentences
        sentence_datas = [self.create_sentence(text) for text in sentences]
        
        # Send all to queue
        async def process_all():
            tasks = [
                self.send_sentence_to_queue(sentence_data)
                for sentence_data in sentence_datas
            ]
            await asyncio.gather(*tasks)
            
            # Wait for labels
            label_tasks = [
                self.wait_and_verify_label(sentence_data['id'])
                for sentence_data in sentence_datas
            ]
            await asyncio.gather(*label_tasks)
        
        asyncio.run(process_all())

if __name__ == "__main__":
    logger.info("Starting Worker Service tests")
    unittest.main(verbosity=2)