import json
import httpx
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentence_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def check_api_health(api_url: str) -> bool:
    """
    Check if the API Gateway and its services are healthy before starting ingestion.
    
    Args:
        api_url (str): Base URL of the API Gateway service
    
    Returns:
        bool: True if all services are healthy, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health_response = await client.get(f"{api_url}/health/services")
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                all_healthy = all(
                    status == "healthy" 
                    for service, status in health_data.items() 
                    if service != "timestamp"
                )
                
                if all_healthy:
                    logger.info("All services are healthy")
                    return True
                else:
                    unhealthy_services = [
                        service for service, status in health_data.items()
                        if service != "timestamp" and status != "healthy"
                    ]
                    logger.warning(f"Unhealthy services detected: {', '.join(unhealthy_services)}")
                    return False
            else:
                logger.error(f"Health check failed with status code: {health_response.status_code}")
                return False
                
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to API Gateway for health check: {str(e)}")
        return False

def load_sentences(file_path: Path) -> List[Dict]:
    """
    Load sentences from a JSON file.
    
    Args:
        file_path (Path): Path to the JSON file containing sentences
    
    Returns:
        List[Dict]: List of sentence dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = json.load(file)
            
        # Validate sentence format
        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise ValueError("Each sentence must be a dictionary")
            if 'id' not in sentence or 'sentence' not in sentence:
                raise ValueError("Each sentence must have 'id' and 'sentence' keys")
            
        logger.info(f"Successfully loaded {len(sentences)} sentences from {file_path}")
        return sentences
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading sentences from {file_path}: {e}")
        raise

async def send_sentence_to_api_gateway(
    sentence: Dict,
    api_url: str,
    client: httpx.AsyncClient
) -> Optional[str]:
    """
    Send a single sentence to the API Gateway service.
    
    Args:
        sentence (Dict): Sentence dictionary with 'id' and 'sentence' keys
        api_url (str): URL of the API Gateway service
        client (httpx.AsyncClient): Async HTTP client
    
    Returns:
        Optional[str]: Sentence ID if successful, None otherwise
    """
    try:
        # Prepare payload according to API gateway's SentenceInput model
        payload = {
            "external_id": str(sentence['id']),  # Ensure string type
            "text": sentence['sentence']
        }
        
        # Send POST request
        response = await client.post(api_url, json=payload)
        
        # Handle different response status codes
        if response.status_code in [201, 202]:
            response_data = response.json()
            sentence_id = response_data.get('id')
            status = response_data.get('status')
            
            if response.status_code == 202:
                logger.warning(
                    f"Sentence {sentence['id']} partially processed. "
                    f"Status: {status}. Message: {response_data.get('message', 'N/A')}"
                )
            else:
                logger.info(
                    f"Successfully processed sentence {sentence['id']}. "
                    f"Assigned ID: {sentence_id}. Status: {status}"
                )
            
            return sentence_id
            
        else:
            logger.error(
                f"Failed to send sentence {sentence['id']}. "
                f"Status code: {response.status_code}. Response: {response.text}"
            )
            return None
    
    except httpx.RequestError as e:
        logger.error(f"Request error for sentence {sentence['id']}: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error processing sentence {sentence['id']}: {e}")
        return None

async def ingest_sentences(file_path: Path, api_url: str, batch_size: int = 10):
    """
    Ingest sentences from a JSON file to the API Gateway service.
    
    Args:
        file_path (Path): Path to the JSON file containing sentences
        api_url (str): URL of the API Gateway service
        batch_size (int): Number of concurrent requests to make
    """
    try:
        # Check API health before starting
        if not await check_api_health(api_url.rsplit('/sentences', 1)[0]):
            logger.error("API Gateway services not healthy. Aborting ingestion.")
            return
        
        # Load sentences
        sentences = load_sentences(file_path)
        
        if not sentences:
            logger.error("No sentences to process.")
            return
        
        logger.info(f"Starting sentence ingestion. Total sentences: {len(sentences)}")
        start_time = time.time()
        
        # Process sentences in batches
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [
                    send_sentence_to_api_gateway(sentence, api_url, client)
                    for sentence in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                successful = sum(1 for r in results if r is not None)
                logger.info(f"Batch {i//batch_size + 1}: {successful}/{len(batch)} sentences processed successfully")
                
                # Rate limiting
                await asyncio.sleep(0.5)  # 500ms between batches
        
        elapsed_time = time.time() - start_time
        logger.info(f"Sentence ingestion completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during sentence ingestion: {e}")
        raise

async def main():
    # Get configuration from environment variables
    json_file_path = Path(os.getenv(
        'SENTENCES_FILE_PATH',
        r'C:\Users\rapha\Desktop\stuff for applications\tasks\voiceLine3\data\knowledgebase.json'
    ))
    api_gateway_url = os.getenv('API_GATEWAY_URL', 'http://localhost:8000/sentences')
    batch_size = int(os.getenv('BATCH_SIZE', '10'))
    
    logger.info(f"Using sentences file: {json_file_path}")
    logger.info(f"API Gateway URL: {api_gateway_url}")
    logger.info(f"Batch size: {batch_size}")
    
    try:
        await ingest_sentences(json_file_path, api_gateway_url, batch_size)
    except Exception as e:
        logger.error(f"Failed to complete sentence ingestion: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())