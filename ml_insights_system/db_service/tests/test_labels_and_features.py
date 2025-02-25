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

    # Test cases for labels with the new sales funnel relevance field
    
    def test_create_label_with_relevance(self):
        """Test creating a label with sales funnel relevance"""
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "is_sales_funnel_relevant": True,
            "is_sales_funnel_relevant_confidence": 0.95,
            "sales_funnel_stage": "Interest",
            "sales_funnel_confidence": 0.85,
            "sentiment": "Positive",
            "sentiment_confidence": 0.92,
            "intent": "Purchase",
            "intent_confidence": 0.78,
            "business_impact": "Medium",
            "business_impact_confidence": 0.65
        }
        
        response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify all fields including the new ones
        self.assertEqual(data["sentence_id"], self.test_sentence["id"])
        self.assertEqual(data["is_sales_funnel_relevant"], label_data["is_sales_funnel_relevant"])
        self.assertEqual(data["is_sales_funnel_relevant_confidence"], label_data["is_sales_funnel_relevant_confidence"])
        self.assertEqual(data["sales_funnel_stage"], label_data["sales_funnel_stage"])
        self.assertEqual(data["sentiment"], label_data["sentiment"])
        logger.info(f"Created label with sales funnel relevance for sentence ID: {self.test_sentence['id']}")

    def test_create_label_not_relevant(self):
        """Test creating a label with sales funnel not relevant"""
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "is_sales_funnel_relevant": False,
            "is_sales_funnel_relevant_confidence": 0.88,
            "sales_funnel_stage": None,  # Should be None for non-relevant
            "sales_funnel_confidence": None,  # Should be None for non-relevant
            "sentiment": "Neutral",
            "sentiment_confidence": 0.75,
            "intent": "Support",
            "intent_confidence": 0.82,
            "business_impact": "Low",
            "business_impact_confidence": 0.70
        }
        
        response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify fields
        self.assertEqual(data["is_sales_funnel_relevant"], False)
        self.assertEqual(data["sales_funnel_stage"], None)
        self.assertEqual(data["sentiment"], "Neutral")
        logger.info(f"Created label with sales funnel not relevant for sentence ID: {self.test_sentence['id']}")

    def test_update_label_relevance(self):
        """Test updating a label's sales funnel relevance"""
        # First create a label without relevance
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "sentiment": "Positive",
            "sentiment_confidence": 0.92
        }
        create_response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Update the label with relevance information
        update_data = {
            "is_sales_funnel_relevant": True,
            "is_sales_funnel_relevant_confidence": 0.87,
            "sales_funnel_stage": "Consideration",
            "sales_funnel_confidence": 0.75
        }
        
        response = requests.patch(
            f"{BASE_URL}/labels/{self.test_sentence['id']}", 
            json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify updates
        self.assertEqual(data["is_sales_funnel_relevant"], update_data["is_sales_funnel_relevant"])
        self.assertEqual(data["is_sales_funnel_relevant_confidence"], update_data["is_sales_funnel_relevant_confidence"])
        self.assertEqual(data["sales_funnel_stage"], update_data["sales_funnel_stage"])
        logger.info(f"Updated label with sales funnel relevance for sentence ID: {self.test_sentence['id']}")

    def test_update_label_from_relevant_to_not_relevant(self):
        """Test updating a label from relevant to not relevant"""
        # First create a label with relevance
        label_data = {
            "sentence_id": self.test_sentence["id"],
            "is_sales_funnel_relevant": True,
            "sales_funnel_stage": "Awareness",
            "sentiment": "Positive"
        }
        create_response = requests.post(f"{BASE_URL}/labels", json=label_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Update to not relevant
        update_data = {
            "is_sales_funnel_relevant": False,
            "is_sales_funnel_relevant_confidence": 0.92,
            "sales_funnel_stage": None,  # Should clear this field
            "sales_funnel_confidence": None  # Should clear this field
        }
        
        response = requests.patch(
            f"{BASE_URL}/labels/{self.test_sentence['id']}", 
            json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify updates - should have cleared the sales funnel stage
        self.assertEqual(data["is_sales_funnel_relevant"], False)
        self.assertIsNone(data["sales_funnel_stage"])
        self.assertIsNone(data["sales_funnel_confidence"])
        # But should keep other fields
        self.assertEqual(data["sentiment"], "Positive")
        logger.info(f"Successfully updated label from relevant to not relevant")

    # Analytics endpoint tests with relevance filtering
    
    def test_analytics_with_relevance_filtering(self):
        """Test analytics endpoints with sales funnel relevance filtering"""
        # Create several test sentences with different relevance values
        
        # Create relevant sentence 1
        relevant_sentence1 = self.create_test_sentence()
        relevant_label1 = {
            "sentence_id": relevant_sentence1["id"],
            "is_sales_funnel_relevant": True,
            "is_sales_funnel_relevant_confidence": 0.95,
            "sales_funnel_stage": "Awareness",
            "sentiment": "Positive",
            "intent": "Information",
            "business_impact": "Low"
        }
        requests.post(f"{BASE_URL}/labels", json=relevant_label1)
        
        # Create relevant sentence 2
        relevant_sentence2 = self.create_test_sentence()
        relevant_label2 = {
            "sentence_id": relevant_sentence2["id"],
            "is_sales_funnel_relevant": True,
            "is_sales_funnel_relevant_confidence": 0.88,
            "sales_funnel_stage": "Interest",
            "sentiment": "Negative",
            "intent": "Purchase",
            "business_impact": "High"
        }
        requests.post(f"{BASE_URL}/labels", json=relevant_label2)
        
        # Create non-relevant sentence
        non_relevant_sentence = self.create_test_sentence()
        non_relevant_label = {
            "sentence_id": non_relevant_sentence["id"],
            "is_sales_funnel_relevant": False,
            "is_sales_funnel_relevant_confidence": 0.91,
            "sales_funnel_stage": None,
            "sentiment": "Neutral",
            "intent": "Support",
            "business_impact": "Medium"
        }
        requests.post(f"{BASE_URL}/labels", json=non_relevant_label)
        
        # Test sentiment distribution endpoint
        sentiment_response = requests.get(f"{BASE_URL}/analytics/sentiment-distribution")
        self.assertEqual(sentiment_response.status_code, 200)
        sentiment_data = sentiment_response.json()
        
        # Only relevant sentences should be in the distribution and grouped by sales funnel stage
        stages = [item["sales_funnel_stage"] for item in sentiment_data]
        self.assertIn("Awareness", stages)
        self.assertIn("Interest", stages)
        
        # Test funnel metrics endpoint
        funnel_response = requests.get(f"{BASE_URL}/analytics/funnel-metrics")
        self.assertEqual(funnel_response.status_code, 200)
        funnel_data = funnel_response.json()
        
        # Only relevant sentences should be counted in funnel metrics
        self.assertEqual(len(funnel_data), 2)  # Should be 2 stages
        
        # Test sentences by filter endpoint - filtering for relevant sentences
        filter_response = requests.get(f"{BASE_URL}/analytics/sentences-by-filter?is_sales_funnel_relevant=true")
        self.assertEqual(filter_response.status_code, 200)
        filter_data = filter_response.json()
        
        # Should only return relevant sentences
        self.assertEqual(len(filter_data), 2)
        filtered_ids = [item["id"] for item in filter_data]
        self.assertIn(relevant_sentence1["id"], filtered_ids)
        self.assertIn(relevant_sentence2["id"], filtered_ids)
        self.assertNotIn(non_relevant_sentence["id"], filtered_ids)
        
        # Test the new sales funnel relevance stats endpoint
        relevance_response = requests.get(f"{BASE_URL}/analytics/sales-funnel-relevance")
        self.assertEqual(relevance_response.status_code, 200)
        relevance_data = relevance_response.json()
        
        # Should show at least 2 relevant and 1 non-relevant
        self.assertGreaterEqual(relevance_data["sales_funnel_relevant"], 2)
        self.assertGreaterEqual(relevance_data["non_sales_funnel_relevant"], 1)
        
        logger.info("Successfully tested analytics endpoints with relevance filtering")

    # Existing test cases below...
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


    def test_create_multiple_features_and_get_all(self):
        """Test creating multiple features and retrieving all of them"""
        # Create multiple test sentences
        sentences = [
            self.create_test_sentence(),
            self.create_test_sentence(),
            self.create_test_sentence()
        ]
        
        # Create features for each sentence
        features_data = [
            {
                "sentence_id": sentences[0]["id"],
                "word_count": 5,
                "char_count": 25,
                "mentions_masterblaster": True,
                "masterblaster_quantity": 2
            },
            {
                "sentence_id": sentences[1]["id"],
                "word_count": 7,
                "char_count": 35,
                "mentions_funpun": True,
                "funpun_quantity": 1
            },
            {
                "sentence_id": sentences[2]["id"],
                "word_count": 6,
                "char_count": 30,
                "mentions_powerpro": True,
                "powerpro_quantity": 3
            }
        ]
        
        # Create features for each sentence
        for feature_data in features_data:
            response = requests.post(f"{BASE_URL}/features", json=feature_data)
            self.assertEqual(response.status_code, 201)
        
        # Retrieve features for each sentence and verify
        for sentence, feature_data in zip(sentences, features_data):
            response = requests.get(f"{BASE_URL}/features/{sentence['id']}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertEqual(data["word_count"], feature_data["word_count"])
            self.assertEqual(data["char_count"], feature_data["char_count"])
        
        logger.info("Successfully created and retrieved multiple feature sets")

    def test_get_features_with_product_mentions(self):
        """Test retrieving features with product mentions"""
        # Create a test sentence
        test_sentence = self.create_test_sentence()
        
        # Create features with multiple product mentions
        feature_data = {
            "sentence_id": test_sentence["id"],
            "word_count": 10,
            "mentions_masterblaster": True,
            "masterblaster_quantity": 2,
            "mentions_funpun": True,
            "funpun_quantity": 1,
            "mentions_powerpro": False
        }
        
        # Create features
        create_response = requests.post(f"{BASE_URL}/features", json=feature_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Retrieve features
        response = requests.get(f"{BASE_URL}/features/{test_sentence['id']}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify product mention details
        self.assertTrue(data["mentions_masterblaster"])
        self.assertEqual(data["masterblaster_quantity"], 2)
        self.assertTrue(data["mentions_funpun"])
        self.assertEqual(data["funpun_quantity"], 1)
        self.assertFalse(data["mentions_powerpro"])
        
        logger.info("Successfully retrieved features with product mentions")

    def test_get_features_with_optional_fields(self):
        """Test creating and retrieving features with optional fields"""
        # Create a test sentence
        test_sentence = self.create_test_sentence()
        
        # Prepare feature data with optional fields
        feature_data = {
            "sentence_id": test_sentence["id"],
            "word_count": 8,
            "char_count": 40,
            "avg_word_length": 5.0,
            "noun_count": 3,
            "verb_count": 2,
            "adj_count": 1,
            "entity_count": 0,
            "embedding_model": "test-model"
        }
        
        # Create features
        create_response = requests.post(f"{BASE_URL}/features", json=feature_data)
        self.assertEqual(create_response.status_code, 201)
        
        # Retrieve features
        response = requests.get(f"{BASE_URL}/features/{test_sentence['id']}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify optional fields
        self.assertEqual(data["word_count"], feature_data["word_count"])
        self.assertEqual(data["char_count"], feature_data["char_count"])
        self.assertEqual(data["avg_word_length"], feature_data["avg_word_length"])
        self.assertEqual(data["noun_count"], feature_data["noun_count"])
        self.assertEqual(data["verb_count"], feature_data["verb_count"])
        self.assertEqual(data["adj_count"], feature_data["adj_count"])
        self.assertEqual(data["entity_count"], feature_data["entity_count"])
        self.assertEqual(data["embedding_model"], feature_data["embedding_model"])
        
        logger.info("Successfully retrieved features with optional fields")

if __name__ == "__main__":
    logger.info("Starting Database Service tests")
    unittest.main(verbosity=2)