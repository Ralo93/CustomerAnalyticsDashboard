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
            "embedding_model", "sentence_embedding",
            # Product mention fields
            "mentions_masterblaster", "masterblaster_quantity",
            "mentions_funpun", "funpun_quantity",
            "mentions_powerpro", "powerpro_quantity"
        ]
        for field in required_fields:
            self.assertIn(field, data)
        
        # Basic validations
        self.assertEqual(data["sentence_id"], sentence["id"])
        self.assertGreater(data["word_count"], 0)
        self.assertGreater(data["char_count"], 0)
        self.assertGreater(data["avg_word_length"], 0)
        
        # No product mentions in this sentence
        self.assertFalse(data["mentions_masterblaster"])
        self.assertFalse(data["mentions_funpun"])
        self.assertFalse(data["mentions_powerpro"])
        
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

    def test_product_mentions_single(self):
        """Test detection of single product mentions without quantities"""
        test_cases = [
            {
                "text": "I'm interested in purchasing a MasterBlaster for my home.",
                "expected": {"mentions_masterblaster": True, "mentions_funpun": False, "mentions_powerpro": False}
            },
            {
                "text": "Can you tell me more about the FunPun product?",
                "expected": {"mentions_masterblaster": False, "mentions_funpun": True, "mentions_powerpro": False}
            },
            {
                "text": "I heard good things about the Power Pro series.",
                "expected": {"mentions_masterblaster": False, "mentions_funpun": False, "mentions_powerpro": True}
            }
        ]
        
        for case in test_cases:
            # Create sentence in the database
            sentence = self.create_sentence(case["text"])
            
            test_data = {
                "sentence_id": sentence["id"],
                "text": sentence["text"]
            }
            
            response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
            self.assertEqual(response.status_code, 201)
            data = response.json()
            
            # Check product mentions match expected
            for product, expected in case["expected"].items():
                self.assertEqual(data[product], expected, f"Failed on '{product}' for text: '{case['text']}'")
            
            # All quantities should be None
            self.assertIsNone(data["masterblaster_quantity"])
            self.assertIsNone(data["funpun_quantity"])
            self.assertIsNone(data["powerpro_quantity"])
        
        logger.info("Product mentions detection test passed")

    def test_product_mentions_with_quantities(self):
        """Test detection of product mentions with quantities"""
        test_cases = [
            {
                "text": "I'd like to order 5 MasterBlasters for my office.",
                "expected": {
                    "mentions_masterblaster": True, 
                    "masterblaster_quantity": 5,
                    "mentions_funpun": False,
                    "mentions_powerpro": False
                }
            },
            {
                "text": "Please ship 10 units of FunPun to our warehouse.",
                "expected": {
                    "mentions_masterblaster": False, 
                    "mentions_funpun": True,
                    "funpun_quantity": 10,
                    "mentions_powerpro": False
                }
            },
            {
                "text": "We need to purchase 25 PowerPro devices as soon as possible.",
                "expected": {
                    "mentions_masterblaster": False, 
                    "mentions_funpun": False,
                    "mentions_powerpro": True,
                    "powerpro_quantity": 25
                }
            },
            {
                "text": "Send me a quote for 3 Master Blaster and 7 Fun Pun items.",
                "expected": {
                    "mentions_masterblaster": True, 
                    "masterblaster_quantity": 3,
                    "mentions_funpun": True,
                    "funpun_quantity": 7,
                    "mentions_powerpro": False
                }
            }
        ]
        
        for case in test_cases:
            # Create sentence in the database
            sentence = self.create_sentence(case["text"])
            
            test_data = {
                "sentence_id": sentence["id"],
                "text": sentence["text"]
            }
            
            response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
            self.assertEqual(response.status_code, 201)
            data = response.json()
            
            # Check product mentions and quantities match expected
            for field, expected in case["expected"].items():
                self.assertEqual(data[field], expected, f"Failed on '{field}' for text: '{case['text']}'")
        
        logger.info("Product mentions with quantities test passed")

    def test_multiple_products_without_quantities(self):
        """Test detection of multiple product mentions without quantities"""
        text = "We currently use both MasterBlaster and PowerPro in our operations."
        sentence = self.create_sentence(text)
        
        test_data = {
            "sentence_id": sentence["id"],
            "text": sentence["text"]
        }
        
        response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Should detect both products
        self.assertTrue(data["mentions_masterblaster"])
        self.assertFalse(data["mentions_funpun"])
        self.assertTrue(data["mentions_powerpro"])
        
        # No quantities
        self.assertIsNone(data["masterblaster_quantity"])
        self.assertIsNone(data["funpun_quantity"])
        self.assertIsNone(data["powerpro_quantity"])
        
        logger.info("Multiple products detection test passed")

    def test_variants_of_product_names(self):
        """Test detection of variants of product names"""
        test_cases = [
            {
                "text": "The Master Blaster is our top seller.",
                "expected": {"mentions_masterblaster": True}
            },
            {
                "text": "Fun Pun toys are very popular with children.",
                "expected": {"mentions_funpun": True}
            },
            {
                "text": "Power Pro is the professional choice.",
                "expected": {"mentions_powerpro": True}
            }
        ]
        
        for case in test_cases:
            # Create sentence in the database
            sentence = self.create_sentence(case["text"])
            
            test_data = {
                "sentence_id": sentence["id"],
                "text": sentence["text"]
            }
            
            response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
            self.assertEqual(response.status_code, 201)
            data = response.json()
            
            # Check product mentions match expected
            for product, expected in case["expected"].items():
                self.assertEqual(data[product], expected, f"Failed on '{product}' for text: '{case['text']}'")
        
        logger.info("Product name variants test passed")

    def test_product_analytics_endpoint(self):
        """Test the product mentions analytics endpoint"""
        # First, create and process several sentences with product mentions
        sentences = [
            "I'd like to order 5 MasterBlasters.",
            "Please ship 10 units of FunPun to our warehouse.",
            "We need to purchase 25 PowerPro devices.",
            "The MasterBlaster and PowerPro are both excellent products.",
            "I'm considering buying a Fun Pun for my nephew."
        ]
        
        # Process each sentence
        for text in sentences:
            sentence = self.create_sentence(text)
            test_data = {
                "sentence_id": sentence["id"],
                "text": sentence["text"]
            }
            response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
            self.assertEqual(response.status_code, 201)
        
        # Now test the analytics endpoint
        response = requests.get(f"{DB_SERVICE_URL}/analytics/product-mentions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Basic validations on the analytics response
        self.assertIn("masterblaster", data)
        self.assertIn("funpun", data)
        self.assertIn("powerpro", data)
        self.assertIn("total_sentences", data)
        self.assertIn("with_any_product_mention", data)
        
        # Check that we have the counts we expect
        self.assertGreaterEqual(data["masterblaster"]["mention_count"], 2)
        self.assertGreaterEqual(data["funpun"]["mention_count"], 2)
        self.assertGreaterEqual(data["powerpro"]["mention_count"], 2)
        
        # For quantities, only count the sentences with explicit quantities.
        # Don't use assertEqual here since other tests might have added to these counts
        self.assertGreaterEqual(data["masterblaster"]["quantity_count"], 5)  
        self.assertGreaterEqual(data["funpun"]["quantity_count"], 10)
        self.assertGreaterEqual(data["powerpro"]["quantity_count"], 25)
        
        logger.info("Product analytics endpoint test passed")

    def test_long_text(self):
        """Test handling of longer text with product mentions"""
        long_text = """
        Our company has been using 15 MasterBlaster devices for the past two years, 
        and we've found them to be extremely reliable. We also purchased 8 FunPun items 
        for our recreation room, which have been popular with visitors. 
        Recently, we upgraded to 20 PowerPro systems for our main production line.
        """
        
        # Create sentence in the database
        sentence = self.create_sentence(long_text)
        
        test_data = {
            "sentence_id": sentence["id"],
            "text": sentence["text"]
        }
        
        response = requests.post(f"{FEATURE_SERVICE_URL}/extract_features", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Check all products and quantities
        self.assertTrue(data["mentions_masterblaster"])
        self.assertEqual(data["masterblaster_quantity"], 15)
        
        self.assertTrue(data["mentions_funpun"])
        self.assertEqual(data["funpun_quantity"], 8)
        
        self.assertTrue(data["mentions_powerpro"])
        self.assertEqual(data["powerpro_quantity"], 20)
        
        logger.info("Long text processing with product mentions test passed")

if __name__ == "__main__":
    logger.info("Starting Feature Service tests")
    unittest.main(verbosity=2)