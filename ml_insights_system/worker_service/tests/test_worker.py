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

    def test_worker_sentence_processing_relevant(self):
        """Test processing for sales funnel relevant sentences"""
        # Test texts that should be recognized as sales funnel relevant
        relevant_texts = [
            "I'm interested in purchasing your premium product",
            "Can you tell me more about your pricing options?",
            "I'd like to compare your service with competitor X",
            "What features are included in the enterprise package?",
            "I'm considering upgrading my subscription"
        ]
        
        for text in relevant_texts:
            # Create sentence
            sentence_data = self.create_sentence(text)
            
            # Run async method to send to queue
            asyncio.run(self.send_sentence_to_queue(sentence_data))
            
            # Wait for processing and verify it's marked as relevant with stage
            asyncio.run(self.wait_and_verify_relevant_label(sentence_data['id']))

    def test_worker_sentence_processing_not_relevant(self):
        """Test processing for non-sales funnel relevant sentences"""
        # Test texts that should be recognized as NOT sales funnel relevant
        non_relevant_texts = [
            "How do I reset my password?",
            "The app crashed when I tried to export data",
            "Thank you for your prompt response",
            "Could you update my email address in your system?",
            "When will the scheduled maintenance be completed?"
        ]
        
        for text in non_relevant_texts:
            # Create sentence
            sentence_data = self.create_sentence(text)
            
            # Run async method to send to queue
            asyncio.run(self.send_sentence_to_queue(sentence_data))
            
            # Wait for processing and verify it's marked as not relevant
            asyncio.run(self.wait_and_verify_not_relevant_label(sentence_data['id']))

    async def wait_and_verify_relevant_label(self, sentence_id, timeout=30):
        """
        Wait for and verify that a sentence is properly classified as sales funnel relevant
        with a sales funnel stage assigned
        
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
                    
                    # Verify it's classified as sales funnel relevant
                    self.assertTrue(
                        label_data['is_sales_funnel_relevant'], 
                        f"Text not marked as sales funnel relevant: {label_data}"
                    )
                    
                    # Check that the confidence score is present
                    self.assertIsNotNone(
                        label_data['is_sales_funnel_relevant_confidence'],
                        "Missing relevance confidence score"
                    )
                    self.assertGreater(
                        label_data['is_sales_funnel_relevant_confidence'], 
                        0.0, 
                        "Relevance confidence score should be greater than 0"
                    )
                    
                    # Verify sales funnel stage is assigned
                    self.assertIsNotNone(
                        label_data['sales_funnel_stage'],
                        "Sales funnel stage not assigned for relevant sentence"
                    )
                    self.assertIsNotNone(
                        label_data['sales_funnel_confidence'],
                        "Sales funnel confidence not assigned for relevant sentence"
                    )
                    
                    # Verify other classifications are also present
                    self.assertIsNotNone(label_data['sentiment'], "Sentiment not processed")
                    self.assertIsNotNone(label_data['intent'], "Intent not processed")
                    self.assertIsNotNone(label_data['business_impact'], "Business impact not processed")
                    
                    logger.info(f"Relevant label successfully created for sentence {sentence_id}")
                    logger.info(f"Stage: {label_data['sales_funnel_stage']}, " +
                               f"Intent: {label_data['intent']}, " +
                               f"Sentiment: {label_data['sentiment']}")
                    return
            
            except Exception as e:
                logger.error(f"Error checking label: {e}")
                await asyncio.sleep(1)
        
        # If we exit the loop, it means we timed out
        self.fail(f"Timed out waiting for label creation for sentence {sentence_id}")

    async def wait_and_verify_not_relevant_label(self, sentence_id, timeout=30):
        """
        Wait for and verify that a sentence is properly classified as NOT sales funnel relevant
        
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
                    
                    # Verify it's classified as NOT sales funnel relevant
                    self.assertFalse(
                        label_data['is_sales_funnel_relevant'], 
                        f"Text incorrectly marked as sales funnel relevant: {label_data}"
                    )
                    
                    # Check that the confidence score is present
                    self.assertIsNotNone(
                        label_data['is_sales_funnel_relevant_confidence'],
                        "Missing relevance confidence score"
                    )
                    
                    # Verify sales funnel stage is NOT assigned (should be None for non-relevant)
                    self.assertIsNone(
                        label_data['sales_funnel_stage'],
                        "Sales funnel stage should be None for non-relevant sentence"
                    )
                    
                    # Verify other classifications are still present despite not being relevant
                    self.assertIsNotNone(label_data['sentiment'], "Sentiment not processed")
                    self.assertIsNotNone(label_data['intent'], "Intent not processed")
                    self.assertIsNotNone(label_data['business_impact'], "Business impact not processed")
                    
                    logger.info(f"Non-relevant label successfully created for sentence {sentence_id}")
                    logger.info(f"Intent: {label_data['intent']}, " +
                               f"Sentiment: {label_data['sentiment']}")
                    return
            
            except Exception as e:
                logger.error(f"Error checking label: {e}")
                await asyncio.sleep(1)
        
        # If we exit the loop, it means we timed out
        self.fail(f"Timed out waiting for label creation for sentence {sentence_id}")

    def test_mixed_sentence_processing(self):
        """Test processing a mix of relevant and non-relevant sentences"""
        # Mix of relevant and non-relevant sentences
        mixed_texts = [
            # Should be relevant
            "What's the pricing for your enterprise tier?",
            # Should not be relevant
            "I need help resetting my account",
            # Should be relevant
            "I'm comparing your product with several competitors",
            # Should not be relevant
            "Thank you for resolving my technical issue"
        ]
        
        # Create sentences
        sentence_datas = [self.create_sentence(text) for text in mixed_texts]
        
        # Send all to queue
        async def process_all():
            # Expected relevance for each text (same order as mixed_texts)
            expected_relevance = [True, False, True, False]
            
            # Send all sentences to queue
            send_tasks = [
                self.send_sentence_to_queue(sentence_data)
                for sentence_data in sentence_datas
            ]
            await asyncio.gather(*send_tasks)
            
            # Wait for labels with appropriate verification
            label_tasks = []
            for i, sentence_data in enumerate(sentence_datas):
                if expected_relevance[i]:
                    # Should be classified as relevant
                    label_tasks.append(self.wait_and_verify_relevant_label(sentence_data['id']))
                else:
                    # Should be classified as not relevant
                    label_tasks.append(self.wait_and_verify_not_relevant_label(sentence_data['id']))
            
            await asyncio.gather(*label_tasks)
        
        asyncio.run(process_all())
        
    def test_edge_case_sentences(self):
        """Test processing of edge case sentences that might be ambiguous"""
        edge_texts = [
            # Ambiguous sentence (could be support or product interest)
            "I'm looking for more information about your product",
            # Very short sentence
            "Help me",
            # Long, complex sentence
            "I've been using your software for 3 years and while I initially loved it, I'm now considering alternatives because your competitors have recently added some features that seem quite useful for my specific use case, though I'd prefer to stay with you if possible.",
            # Mixed intent
            "I like your product but I'm having trouble with the login page"
        ]
        
        for text in edge_texts:
            # Create sentence
            sentence_data = self.create_sentence(text)
            
            # Run async method to send to queue
            asyncio.run(self.send_sentence_to_queue(sentence_data))
            
            # Just verify it gets processed (without asserting whether it should be relevant or not)
            asyncio.run(self.wait_and_verify_label(sentence_data['id']))

    async def wait_and_verify_label(self, sentence_id, timeout=30):
        """
        Wait for and verify label creation (basic version without relevance checks)
        
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
                    
                    # Verify relevance classification is present
                    self.assertIsNotNone(
                        label_data['is_sales_funnel_relevant'], 
                        "Sales funnel relevance not classified"
                    )
                    self.assertIsNotNone(
                        label_data['is_sales_funnel_relevant_confidence'], 
                        "Sales funnel relevance confidence not set"
                    )
                    
                    # Verify other classifications are present
                    self.assertIsNotNone(label_data['sentiment'], "Sentiment not processed")
                    self.assertIsNotNone(label_data['intent'], "Intent not processed")
                    self.assertIsNotNone(label_data['business_impact'], "Business impact not processed")
                    
                    # If relevant, check for sales funnel stage
                    if label_data['is_sales_funnel_relevant']:
                        self.assertIsNotNone(
                            label_data['sales_funnel_stage'], 
                            "Sales funnel stage missing for relevant sentence"
                        )
                    else:
                        # Not relevant - stage should be None
                        self.assertIsNone(
                            label_data['sales_funnel_stage'],
                            "Sales funnel stage should be None for non-relevant sentence"
                        )
                    
                    logger.info(f"Label successfully created for sentence {sentence_id}")
                    logger.info(f"Relevant: {label_data['is_sales_funnel_relevant']}, " + 
                               (f"Stage: {label_data['sales_funnel_stage']}, " if label_data['is_sales_funnel_relevant'] else "") +
                               f"Intent: {label_data['intent']}, " +
                               f"Sentiment: {label_data['sentiment']}")
                    return
            
            except Exception as e:
                logger.error(f"Error checking label: {e}")
                await asyncio.sleep(1)
        
        # If we exit the loop, it means we timed out
        self.fail(f"Timed out waiting for label creation for sentence {sentence_id}")

if __name__ == "__main__":
    logger.info("Starting Worker Service tests")
    unittest.main(verbosity=2)