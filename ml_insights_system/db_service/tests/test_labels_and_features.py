import base64
import requests
import unittest
import json
import uuid
import logging
import numpy as np
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Service URL
BASE_URL = "http://localhost:8001"

class DatabaseServiceTests(unittest.TestCase):
    
    def setUp(self):
        """Setup before each test - verify service is running"""
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                self.fail("Service is not running. Please start the service before running tests.")
            logger.info("Service is running. Starting tests.")
            
            # Create a test sentence to use in feature and label tests
            self.test_sentence = self.create_test_sentence()
            
        except requests.ConnectionError:
            self.fail("Could not connect to the service. Please start the service before running tests.")
    
    def create_test_sentence(self):
        """Helper method to create a test sentence"""
        test_data = {
            "external_id": str(uuid.uuid4()),
            "text": "This is a test sentence for features and labels"
        }
        response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_create_sentence(self):
        """Test creating a new sentence"""
        test_data = {
            "external_id": str(uuid.uuid4()),
            "text": "Test sentence creation"
        }
        response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify fields
        self.assertEqual(data["text"], test_data["text"])
        self.assertEqual(data["external_id"], test_data["external_id"])
        self.assertTrue("id" in data)
        self.assertTrue("created_at" in data)
        logger.info(f"Created sentence with ID: {data['id']}")

    def test_get_sentences(self):
        """Test retrieving all sentences"""
        response = requests.get(f"{BASE_URL}/sentences")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        logger.info(f"Retrieved {len(data)} sentences")

    def test_get_sentence(self):
        """Test retrieving a specific sentence"""
        response = requests.get(f"{BASE_URL}/sentences/{self.test_sentence['id']}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], self.test_sentence["id"])
        self.assertEqual(data["text"], self.test_sentence["text"])
        logger.info(f"Retrieved sentence {self.test_sentence['id']}")

    def test_update_sentence(self):
        """Test updating a sentence"""
        update_data = {
            "text": "Updated test sentence"
        }
        response = requests.patch(
            f"{BASE_URL}/sentences/{self.test_sentence['id']}", 
            json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], update_data["text"])
        logger.info(f"Updated sentence {self.test_sentence['id']}")

    def test_create_features_with_embedding(self):
        """Test creating features with embedding"""
        # Create a random 384-dimensional embedding
        embedding = np.random.rand(384)
        encoded_embedding = base64.b64encode(embedding.tobytes()).decode()
        
        feature_data = {
            "sentence_id": self.test_sentence["id"],
            "word_count": 9,
            "char_count": 45,
            "avg_word_length": 4.2,
            "noun_count": 3,
            "verb_count": 1,
            "adj_count": 1,
            "entity_count": 0,
            "sentence_embedding": encoded_embedding,
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
        response = requests.post(f"{BASE_URL}/features", json=feature_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify essential fields
        self.assertEqual(data["sentence_id"], self.test_sentence["id"])
        self.assertEqual(data["word_count"], feature_data["word_count"])
        self.assertEqual(data["embedding_model"], feature_data["embedding_model"])
        
        # Verify embedding is returned as base64
        self.assertIsNotNone(data["sentence_embedding"])
        # Try decoding the embedding to verify it's valid base64
        decoded_embedding = np.frombuffer(
            base64.b64decode(data["sentence_embedding"]), 
            dtype=np.float64
        )
        self.assertEqual(len(decoded_embedding), 384)
        
        logger.info(f"Created features with embedding for sentence ID: {self.test_sentence['id']}")

    def test_create_features_with_invalid_embedding_dimension(self):
        """Test creating features with invalid embedding dimension"""
        # Create a wrong-sized embedding (e.g., 100-dimensional)
        embedding = np.random.rand(100)
        encoded_embedding = base64.b64encode(embedding.tobytes()).decode()
        
        feature_data = {
            "sentence_id": self.test_sentence["id"],
            "word_count": 5,
            "sentence_embedding": encoded_embedding
        }
        
        response = requests.post(f"{BASE_URL}/features", json=feature_data)
        self.assertEqual(response.status_code, 422)  # Validation error
        self.assertIn("384-dimensional", response.json()["detail"][0]["msg"].lower())
        
        logger.info("Successfully rejected invalid embedding dimension")

    def test_update_features_with_embedding(self):
        """Test updating features with new embedding"""
        # First create features without embedding
        initial_feature_data = {
            "sentence_id": self.test_sentence["id"],
            "word_count": 5,
        }
        create_response = requests.post(f"{BASE_URL}/features", json=initial_feature_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Create new embedding for update
        new_embedding = np.random.rand(384)
        encoded_embedding = base64.b64encode(new_embedding.tobytes()).decode()
        
        # Update features with new embedding
        update_data = {
            "word_count": 6,
            "sentence_embedding": encoded_embedding
        }
        
        response = requests.patch(
            f"{BASE_URL}/features/{self.test_sentence['id']}", 
            json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify updates
        self.assertEqual(data["word_count"], update_data["word_count"])
        
        # Verify embedding
        self.assertIsNotNone(data["sentence_embedding"])
        decoded_embedding = np.frombuffer(
            base64.b64decode(data["sentence_embedding"]), 
            dtype=np.float64
        )
        self.assertEqual(len(decoded_embedding), 384)
        
        logger.info(f"Updated features with new embedding for sentence ID: {self.test_sentence['id']}")

    def test_create_features_without_embedding(self):
        """Test creating features without embedding"""
        feature_data = {
            "sentence_id": self.test_sentence["id"],
            "word_count": 5,
        }
        
        response = requests.post(f"{BASE_URL}/features", json=feature_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify fields
        self.assertEqual(data["sentence_id"], self.test_sentence["id"])
        self.assertEqual(data["word_count"], feature_data["word_count"])
        self.assertIsNone(data["sentence_embedding"])
        
        logger.info(f"Created features without embedding for sentence ID: {self.test_sentence['id']}")

    def test_get_features_with_embedding(self):
        """Test retrieving features with embedding"""
        # First create features with embedding
        embedding = np.random.rand(384)
        encoded_embedding = base64.b64encode(embedding.tobytes()).decode()
        
        feature_data = {
            "sentence_id": self.test_sentence["id"],
            "word_count": 5,
            "sentence_embedding": encoded_embedding
        }
        
        create_response = requests.post(f"{BASE_URL}/features", json=feature_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Now retrieve them
        response = requests.get(f"{BASE_URL}/features/{self.test_sentence['id']}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify fields
        self.assertEqual(data["word_count"], feature_data["word_count"])
        
        # Verify embedding
        self.assertIsNotNone(data["sentence_embedding"])
        decoded_embedding = np.frombuffer(
            base64.b64decode(data["sentence_embedding"]), 
            dtype=np.float64
        )
        self.assertEqual(len(decoded_embedding), 384)
        
        logger.info(f"Retrieved features with embedding for sentence ID: {self.test_sentence['id']}")

    def test_create_label(self):
        """Test creating a label for a sentence"""
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "sales_funnel_stage": "prospect",
            "sales_funnel_confidence": 0.85,
            "sentiment": "positive",
            "sentiment_confidence": 0.92,
            "intent": "inquiry",
            "intent_confidence": 0.78,
            "business_impact": "medium",
            "business_impact_confidence": 0.65
        }
        
        response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        self.assertEqual(data["sentence_id"], self.test_sentence["id"])
        self.assertEqual(data["sales_funnel_stage"], label_data["sales_funnel_stage"])
        self.assertEqual(data["sentiment"], label_data["sentiment"])
        logger.info(f"Created label for sentence ID: {self.test_sentence['id']}")

    def test_create_duplicate_label(self):
        """Test that creating duplicate labels raises an error"""
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "sentiment": "positive",
            "sentiment_confidence": 0.92
        }
        
        # First creation should succeed
        response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(response.status_code, 201)
        
        # Second creation should fail
        response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("already exists", response.json()["detail"])
        logger.info("Correctly prevented duplicate label creation")

    def test_get_label(self):
        """Test retrieving a label for a sentence"""
        # First create a label
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "sentiment": "positive",
            "sentiment_confidence": 0.92
        }
        create_response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Now retrieve it
        response = requests.get(f"{BASE_URL}/labels/{self.test_sentence['id']}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["sentiment"], label_data["sentiment"])
        self.assertEqual(data["sentiment_confidence"], label_data["sentiment_confidence"])
        logger.info(f"Retrieved label for sentence ID: {self.test_sentence['id']}")

    def test_update_label(self):
        """Test updating a label for a sentence"""
        # First create a label
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "sentiment": "positive",
            "sentiment_confidence": 0.92
        }
        create_response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Record the initial last_updated time
        initial_timestamp = create_response.json()["last_updated"]
        
        # Wait a moment to ensure timestamp would be different
        time.sleep(0.1)
        
        # Update the label
        update_data = {
            "sentiment": "negative",
            "sentiment_confidence": 0.88,
            "sales_funnel_stage": "lead"
        }
        response = requests.patch(
            f"{BASE_URL}/labels/{self.test_sentence['id']}", 
            json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify updates
        self.assertEqual(data["sentiment"], update_data["sentiment"])
        self.assertEqual(data["sentiment_confidence"], update_data["sentiment_confidence"])
        self.assertEqual(data["sales_funnel_stage"], update_data["sales_funnel_stage"])
        
        # Verify last_updated was changed
        self.assertNotEqual(data["last_updated"], initial_timestamp)
        logger.info(f"Updated label for sentence ID: {self.test_sentence['id']}")

    def test_nonexistent_resources(self):
        """Test operations with non-existent IDs"""
        fake_id = str(uuid.uuid4())
        
        # Test sentence endpoints
        self.assertEqual(
            requests.get(f"{BASE_URL}/sentences/{fake_id}").status_code, 
            404
        )
        self.assertEqual(
            requests.patch(
                f"{BASE_URL}/sentences/{fake_id}", 
                json={"text": "test"}
            ).status_code, 
            404
        )
        
        # Test features endpoints
        self.assertEqual(
            requests.get(f"{BASE_URL}/features/{fake_id}").status_code, 
            404
        )
        self.assertEqual(
            requests.patch(
                f"{BASE_URL}/features/{fake_id}", 
                json={"word_count": 5}
            ).status_code, 
            404
        )
        self.assertEqual(
            requests.post(
                f"{BASE_URL}/features", 
                json={"sentence_id": fake_id}
            ).status_code, 
            404
        )
        
        # Test labels endpoints
        self.assertEqual(
            requests.get(f"{BASE_URL}/labels/{fake_id}").status_code, 
            404
        )
        self.assertEqual(
            requests.patch(
                f"{BASE_URL}/labels/{fake_id}", 
                json={"sentiment": "positive"}
            ).status_code, 
            404
        )
        self.assertEqual(
            requests.post(
                f"{BASE_URL}/labels", 
                json={"sentence_id": fake_id}
            ).status_code, 
            404
        )
        
        logger.info("Successfully tested operations with non-existent IDs")

if __name__ == "__main__":
    logger.info("Starting Database Service tests")
    unittest.main(verbosity=2)