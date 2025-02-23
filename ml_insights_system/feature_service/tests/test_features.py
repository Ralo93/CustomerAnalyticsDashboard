import unittest
import requests
import uuid
import logging
import numpy as np
import base64
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service URLs
FEATURE_SERVICE_URL = "http://localhost:8003"
DB_SERVICE_URL = "http://localhost:8001"

class FeatureServiceTests(unittest.TestCase):
    
    def create_sentence(self, text):
        """Helper method to create a sentence in the database service"""
        sentence_data = {
            "external_id": str(uuid.uuid4()),
            "text": text
        }
        response = requests.post(f"{DB_SERVICE_URL}/sentences", json=sentence_data)
        self.assertEqual(response.status_code, 201)
        return response.json()

    def setUp(self):
        """Setup before each test - verify services are running"""
        try:
            # Check feature service
            feature_response = requests.get(f"{FEATURE_SERVICE_URL}/health")
            if feature_response.status_code != 200:
                self.fail("Feature service is not running.")
            
            # Check database service
            db_response = requests.get(f"{DB_SERVICE_URL}/health")
            if db_response.status_code != 200:
                self.fail("Database service is not running.")
            
            logger.info("Both services are running. Starting tests.")
            
        except requests.ConnectionError:
            self.fail("Could not connect to services. Please start both services before running tests.")
    
    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        # Create sentence in the database first
        sentence = self.create_sentence("This is a simple test sentence with some nouns and verbs.")
        
        # Use the created sentence's ID for feature extraction
        test_data = {
            "sentence_id": sentence["id"],
            "text": sentence["text"]
        }
        
        logger.debug(f"Sending request to {FEATURE_SERVICE_URL}/extract_features with data: {test_data}")
        response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
        
        if response.status_code != 201:
            logger.error(f"Response: {response.text}")
            
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Check all required fields are present
        required_fields = [
            "sentence_id", "word_count", "char_count", "avg_word_length",
            "noun_count", "verb_count", "adj_count", "entity_count",
            "embedding_model", "sentence_embedding"
        ]
        for field in required_fields:
            self.assertIn(field, data)
        
        # Basic validations
        self.assertEqual(data["sentence_id"], sentence["id"])
        self.assertGreater(data["word_count"], 0)
        self.assertGreater(data["char_count"], 0)
        self.assertGreater(data["avg_word_length"], 0)
        
        logger.info("Basic feature extraction test passed")

    def test_extract_features_with_entities(self):
        """Test feature extraction with named entities"""
        # Create sentence in the database first
        sentence = self.create_sentence("Microsoft and Google are working on AI projects in New York.")
        
        test_data = {
            "sentence_id": sentence["id"],
            "text": sentence["text"]
        }
        
        response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Should detect organizations and location
        self.assertGreater(data["entity_count"], 2)
        
        logger.info("Entity detection test passed")

    def test_long_text(self):
        """Test handling of longer text"""
        long_text = " ".join(["This is a test sentence."] * 50)
        
        # Create sentence in the database first
        sentence = self.create_sentence(long_text)
        
        test_data = {
            "sentence_id": sentence["id"],
            "text": sentence["text"]
        }
        
        response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Basic validations for longer text
        self.assertGreater(data["word_count"], 200)
        self.assertGreater(data["char_count"], 1000)
        
        logger.info("Long text processing test passed")

if __name__ == "__main__":
    logger.info("Starting Feature Service tests")
    unittest.main(verbosity=2)