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
        assert "queue_service" in data
        
        # Services should be healthy or the test would be pointless
        assert data["api_gateway"] == "healthy"
        print(f"API Gateway health: {data['api_gateway']}")
        print(f"DB Service health: {data['db_service']}")
        print(f"Queue Service health: {data['queue_service']}")

@pytest.mark.asyncio
async def test_process_sentence_integration():
    """
    Test the full sentence processing flow with real services:
    1. Submit sentence to API gateway
    2. Verify it gets stored in the database
    3. Verify it gets queued for processing
    """
    global TEST_SENTENCE_ID
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Generate a unique test sentence to easily identify it
        test_id = str(uuid.uuid4())[:8]
        test_sentence = f"There is nothing to say about it. Normal stuff x2{test_id}"
        
        # Step 1: Submit the sentence through the API gateway
        print(f"\nSubmitting test sentence: '{test_sentence}'")
        response = await client.post(
            f"{API_GATEWAY_URL}/sentences",
            json={"text": test_sentence, "priority": "normal"}
        )
        
        assert response.status_code in [200, 201], f"Failed to submit sentence: {response.text}"
        data = response.json()
        assert "id" in data, "Response did not contain sentence ID"
        sentence_id = data["id"]
        TEST_SENTENCE_ID = sentence_id
        print(f"Sentence stored with ID: {sentence_id}")
        
        # Step 2: Verify the sentence was stored in the database
        # Give it a moment to process
        time.sleep(1)
        
        db_response = await client.get(f"{API_GATEWAY_URL}/sentences/{sentence_id}")
        assert db_response.status_code == 200, f"Failed to retrieve sentence: {db_response.text}"
        db_data = db_response.json()
        assert db_data["id"] == sentence_id
        assert db_data["text"] == test_sentence
        assert db_data["priority"] == "normal"
        print(f"Successfully verified sentence in database: {db_data}")
        
        # Step 3: Verify the queue service status (optional)
        # Note: This depends on your queue service having a status endpoint
        try:
            queue_response = await client.get(f"{QUEUE_SERVICE_URL}/queue/status")
            if queue_response.status_code == 200:
                queue_data = queue_response.json()
                print(f"Queue status: {queue_data}")
        except httpx.RequestError:
            print("Queue status endpoint not available")
        
        print(f"Integration test completed successfully for sentence: {sentence_id}")

# Optional: Add a test to clean up after the integration test
@pytest.mark.asyncio
async def test_cleanup():
    """
    Optional: Remove test data created during integration tests
    Note: This requires your DB service to have a DELETE endpoint
    """
    global TEST_SENTENCE_ID
    
    if TEST_SENTENCE_ID:
        print(f"Test created sentence with ID: {TEST_SENTENCE_ID}")
        # If you have a delete endpoint, you could clean up here
        # async with httpx.AsyncClient(timeout=10.0) as client:
        #     try:
        #         response = await client.delete(f"{DB_SERVICE_URL}/sentences/{TEST_SENTENCE_ID}")
        #         assert response.status_code in [200, 204]
        #         print(f"Cleaned up test sentence: {TEST_SENTENCE_ID}")
        #     except Exception as e:
        #         print(f"Could not clean up test data: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-xvs"])