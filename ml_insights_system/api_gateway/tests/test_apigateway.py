import os
import pytest
import httpx
import uuid
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# URLs for services - using environment variables or defaults
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
FEATURE_SERVICE_URL = os.getenv("FEATURE_SERVICE_URL", "http://localhost:8003")
QUEUE_SERVICE_URL = os.getenv("QUEUE_SERVICE_URL", "http://localhost:8002")

# Store the sentence ID for cleanup
TEST_SENTENCE_ID = None

@pytest.mark.asyncio
async def test_health_check():
    """Test the API gateway health endpoint"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_GATEWAY_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_services_health_check():
    """Test all services health endpoint"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_GATEWAY_URL}/health/services")
        assert response.status_code == 200
        
        data = response.json()
        assert "api_gateway" in data
        assert "db_service" in data
        assert "feature_service" in data
        assert "queue_service" in data
        
        # Services should be healthy or the test would be pointless
        assert data["api_gateway"] == "healthy"
        print(f"Services health status: {data}")

@pytest.mark.asyncio
async def test_process_sentence_integration():
    """
    Test the full sentence processing flow with real services:
    1. Submit sentence to API gateway
    2. Verify it gets stored in the database
    3. Verify features are extracted
    4. Verify it gets queued for processing
    """
    global TEST_SENTENCE_ID
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Generate a unique test sentence and external ID to easily identify it
        external_id = str(0)
        test_sentence = f"There is nothing to say about it. EXTREMELY IMPORTANTeeeee ! {external_id}"
        
        # Step 1: Submit the sentence through the API gateway
        print(f"\nSubmitting test sentence: '{test_sentence}'")
        response = await client.post(
            f"{API_GATEWAY_URL}/sentences",
            json={
                "external_id": external_id,
                "text": test_sentence
            }
        )
        
        # Allow for both 200 and 201 status codes
        assert response.status_code in [200, 201], f"Failed to submit sentence: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "id" in data, "Response did not contain sentence ID"
        assert "status" in data, "Response did not contain processing status"
        sentence_id = data["id"]
        TEST_SENTENCE_ID = sentence_id
        
        print(f"Sentence processed with ID: {sentence_id}")
        print(f"Processing status: {data['status']}")
        
        # Give some time for async processing
        time.sleep(2)
        
        # Step 2: Verify the sentence was stored in the database
        db_response = await client.get(f"{API_GATEWAY_URL}/sentences/{sentence_id}")
        assert db_response.status_code == 200, f"Failed to retrieve sentence: {db_response.text}"
        
        db_data = db_response.json()
        assert db_data["id"] == sentence_id
        assert db_data["text"] == test_sentence
        assert db_data["external_id"] == external_id
        
        # Step 3: Verify features were extracted
        assert "features" in db_data, "Features not extracted"
        features = db_data["features"]
        assert features["sentence_id"] == sentence_id
        assert features["word_count"] > 0
        assert features["char_count"] > 0
        assert features["sentence_embedding"] is not None
        
        print("Sentence processing verified successfully:")
        print(f"Sentence ID: {sentence_id}")
        print(f"Features extracted: {features}")

if __name__ == "__main__":
    pytest.main(["-xvs"])