import requests
import unittest
import json
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Service URL
BASE_URL = "http://localhost:8001"  # Update if your service runs on a different port


class DatabaseServiceTests(unittest.TestCase):
    
    def setUp(self):
        """Setup before each test - verify service is running"""
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                self.fail("Service is not running. Please start the service before running tests.")
            logger.info("Service is running. Starting tests.")
        except requests.ConnectionError:
            self.fail("Could not connect to the service. Please start the service before running tests.")
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "database_service")
        logger.info("Health check test passed")
    
    def test_create_sentence_basic(self):
        """Test creating a sentence with basic data"""
        test_data = {
            "text": "This is a test sentence",
            "priority": "normal"
        }
        response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["text"], test_data["text"])
        self.assertEqual(data["priority"], test_data["priority"])
        self.assertIsNotNone(data["id"])
        
        # Store the ID for later tests
        self.sentence_id = data["id"]
        logger.info(f"Created sentence with ID: {self.sentence_id}")
    
    def test_create_sentence_full(self):
        """Test creating a sentence with all fields"""
        test_data = {
            "text": "This is a full test sentence with all fields",
            "priority": "high",
            "intent_type": "question",
            "sentiment": "positive",
            "word_count": 9,
            "processing_status": "pending",
            "priority_reasoning": "Contains urgent keywords"
        }
        
        response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify all fields were saved correctly
        for key, value in test_data.items():
            self.assertEqual(data[key], value)
        
        # Store the ID for later tests
        self.full_sentence_id = data["id"]
        logger.info(f"Created full sentence with ID: {self.full_sentence_id}")
    
    def test_get_sentence(self):
        """Test retrieving a sentence"""
        # First create a sentence
        test_data = {"text": "Sentence for get test", "priority": "normal"}
        create_response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(create_response.status_code, 201)
        sentence_id = create_response.json()["id"]
        
        # Now retrieve it
        response = requests.get(f"{BASE_URL}/sentences/{sentence_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], test_data["text"])
        self.assertEqual(data["id"], sentence_id)
        logger.info(f"Retrieved sentence with ID: {sentence_id}")
    
    def test_get_nonexistent_sentence(self):
        """Test retrieving a sentence that doesn't exist"""
        fake_id = str(uuid.uuid4())
        response = requests.get(f"{BASE_URL}/sentences/{fake_id}")
        self.assertEqual(response.status_code, 404)
        logger.info("Correctly got 404 for non-existent sentence")
    
    def test_update_sentence(self):
        """Test updating a sentence"""
        # First create a sentence
        test_data = {"text": "Sentence for update test", "priority": "normal"}
        create_response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(create_response.status_code, 201)
        sentence_id = create_response.json()["id"]
        
        # Now update it
        update_data = {
            "priority": "high",
            "intent_type": "command",
            "sentiment": "neutral",
            "processing_status": "processed",
            "priority_reasoning": "Updated reasoning"
        }
        
        response = requests.patch(f"{BASE_URL}/sentences/{sentence_id}", json=update_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify the updated fields
        for key, value in update_data.items():
            self.assertEqual(data[key], value)
        
        # Original text should remain unchanged
        self.assertEqual(data["text"], test_data["text"])
        logger.info(f"Updated sentence with ID: {sentence_id}")
    
    def test_update_nonexistent_sentence(self):
        """Test updating a sentence that doesn't exist"""
        fake_id = str(uuid.uuid4())
        update_data = {"priority": "high"}
        response = requests.patch(f"{BASE_URL}/sentences/{fake_id}", json=update_data)
        self.assertEqual(response.status_code, 404)
        logger.info("Correctly got 404 for updating non-existent sentence")
    
    def test_priority_values(self):
        """Test setting different priority values"""
        priority_values = ["high", "medium", "low", "urgent", "custom-priority"]
        
        for priority in priority_values:
            test_data = {"text": f"Priority test with {priority}", "priority": priority}
            response = requests.post(f"{BASE_URL}/sentences", json=test_data)
            self.assertEqual(response.status_code, 201)
            data = response.json()
            # Priority should be saved as provided since it's now a string
            self.assertEqual(data["priority"], priority)
            logger.info(f"Correctly set priority to '{priority}'")
    
    def test_custom_priority_update(self):
        """Test updating with custom priority values"""
        # First create a sentence
        test_data = {"text": "Sentence for custom priority update", "priority": "normal"}
        create_response = requests.post(f"{BASE_URL}/sentences", json=test_data)
        self.assertEqual(create_response.status_code, 201)
        sentence_id = create_response.json()["id"]
        
        # Update with a custom priority
        custom_priority = "super-urgent-critical"
        update_data = {"priority": custom_priority}
        
        response = requests.patch(f"{BASE_URL}/sentences/{sentence_id}", json=update_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify the custom priority was saved
        self.assertEqual(data["priority"], custom_priority)
        logger.info(f"Successfully updated with custom priority '{custom_priority}'")


if __name__ == "__main__":
    logger.info("Starting Database Service tests")
    unittest.main(verbosity=2)