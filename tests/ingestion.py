import json
import requests
import time
import logging
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

def load_sentences(file_path):
    """
    Load sentences from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing sentences
    
    Returns:
        list: List of sentence dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = json.load(file)
        return sentences
    except Exception as e:
        logger.error(f"Error loading sentences from {file_path}: {e}")
        return []

def send_sentence_to_api_gateway(sentence, api_url):
    """
    Send a single sentence to the API Gateway service.
    
    Args:
        sentence (dict): Sentence dictionary with 'id' and 'sentence' keys
        api_url (str): URL of the API Gateway service
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare payload
        payload = {
            "id": sentence.get('id'),
            "text": sentence.get('sentence')
        }
        
        # Send POST request
        response = requests.post(api_url, json=payload, timeout=10)
        
        # Check response
        if response.status_code in [200, 201]:
            logger.info(f"Successfully sent sentence {sentence['id']}: {sentence['sentence']}")
            return True
        else:
            logger.error(f"Failed to send sentence {sentence['id']}. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    
    except requests.RequestException as e:
        logger.error(f"Request error for sentence {sentence['id']}: {e}")
        return False

def ingest_sentences(file_path, api_url):
    """
    Ingest sentences from a JSON file to the API Gateway service.
    
    Args:
        file_path (str): Path to the JSON file containing sentences
        api_url (str): URL of the API Gateway service
    """
    # Load sentences
    sentences = load_sentences(file_path)
    
    if not sentences:
        logger.error("No sentences to process.")
        return
    
    logger.info(f"Starting sentence ingestion. Total sentences: {len(sentences)}")
    
    # Process sentences one by one
    for sentence in sentences:
        try:
            # Send sentence
            success = send_sentence_to_api_gateway(sentence, api_url)
            
            # Sleep between requests
            time.sleep(0.2)
        
        except Exception as e:
            logger.error(f"Unexpected error processing sentence {sentence.get('id')}: {e}")

def main():
    # Get configuration from environment variables
    json_file_path = r'C:\Users\rapha\Desktop\stuff for applications\tasks\voiceLine3\data\knowledgebase.json'
    api_gateway_url = os.getenv('API_GATEWAY_URL', 'http://localhost:8000/sentences')
    
    logger.info(f"Using sentences file: {json_file_path}")
    logger.info(f"API Gateway URL: {api_gateway_url}")
    
    # Ingest sentences
    ingest_sentences(json_file_path, api_gateway_url)
    
    logger.info("Sentence ingestion completed.")

if __name__ == '__main__':
    main()