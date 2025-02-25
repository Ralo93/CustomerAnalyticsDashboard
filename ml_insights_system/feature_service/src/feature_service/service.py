from fastapi import FastAPI, HTTPException, status
import logging
import httpx
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field
import spacy
import os
from sentence_transformers import SentenceTransformer
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see more information
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Feature Extraction Service")

# Initialize models
try:
    nlp = spacy.load("en_core_web_sm")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# Configuration
DB_SERVICE_URL = os.getenv('DB_SERVICE_URL', 'http://localhost:8001')
PORT = int(os.getenv('PORT', 8003))

class FeatureExtractionRequest(BaseModel):
    sentence_id: str = Field(..., description="UUID of the sentence")
    text: str = Field(..., min_length=1, description="Text to analyze")
    
class FeatureExtractionResponse(BaseModel):
    sentence_id: str
    word_count: int
    char_count: int
    avg_word_length: float
    noun_count: int
    verb_count: int
    adj_count: int
    entity_count: int
    embedding_model: str
    sentence_embedding: Optional[str] = None
    
    # Product mention fields
    mentions_masterblaster: bool = False
    masterblaster_quantity: Optional[int] = None
    mentions_funpun: bool = False
    funpun_quantity: Optional[int] = None
    mentions_powerpro: bool = False
    powerpro_quantity: Optional[int] = None

def extract_product_mentions(text):
    """
    Extract product mentions and their quantities from text
    
    Returns:
        dict: Dictionary with product mentions and quantities
    """
    result = {
        "mentions_masterblaster": False,
        "masterblaster_quantity": None,
        "mentions_funpun": False,
        "funpun_quantity": None,
        "mentions_powerpro": False,
        "powerpro_quantity": None
    }
    
    text_lower = text.lower()
    
    # Check for MasterBlaster mentions
    if "masterblaster" in text_lower or "master blaster" in text_lower:
        result["mentions_masterblaster"] = True
        # Try to find quantity using regex
        import re
        quantity_match = re.search(r'(\d+)\s*(?:units?|pieces?|pcs?|qty|quantity)?\s*(?:of)?\s*(?:masterblasters?|master\s+blasters?)', text_lower)
        if quantity_match:
            result["masterblaster_quantity"] = int(quantity_match.group(1))
    
    # Check for FunPun mentions
    if "funpun" in text_lower or "fun pun" in text_lower:
        result["mentions_funpun"] = True
        # Try to find quantity
        import re
        quantity_match = re.search(r'(\d+)\s*(?:units?|pieces?|pcs?|qty|quantity)?\s*(?:of)?\s*(?:funpuns?|fun\s+puns?)', text_lower)
        if quantity_match:
            result["funpun_quantity"] = int(quantity_match.group(1))
    
    # Check for PowerPro mentions
    if "powerpro" in text_lower or "power pro" in text_lower:
        result["mentions_powerpro"] = True
        # Try to find quantity
        import re
        quantity_match = re.search(r'(\d+)\s*(?:units?|pieces?|pcs?|qty|quantity)?\s*(?:of)?\s*(?:powerpros?|power\s+pros?)', text_lower)
        if quantity_match:
            result["powerpro_quantity"] = int(quantity_match.group(1))
    
    return result
@app.post(
    "/extract_features", 
    response_model=FeatureExtractionResponse,
    status_code=status.HTTP_201_CREATED
)
async def extract_features(request: FeatureExtractionRequest):
    """Extract features from text and store them in the database"""
    logger.info(f"Processing sentence ID: {request.sentence_id}")
    
    try:
        # Process text with spaCy
        doc = nlp(request.text)
        
        # Extract basic features
        word_count = len([token for token in doc if not token.is_punct])
        char_count = len(request.text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Count POS tags
        noun_count = len([token for token in doc if token.pos_ == "NOUN"])
        verb_count = len([token for token in doc if token.pos_ == "VERB"])
        adj_count = len([token for token in doc if token.pos_ == "ADJ"])
        
        # Count named entities
        entity_count = len(doc.ents)
        
        # Extract product mentions
        product_mentions = extract_product_mentions(request.text)
        
        # Generate embedding
        embedding = embedding_model.encode(request.text)
        embedding_float64 = embedding.astype(np.float64)
        encoded_embedding = base64.b64encode(embedding_float64.tobytes()).decode()
        
        # Prepare feature data
        feature_data = {
            "sentence_id": request.sentence_id,
            "word_count": word_count,
            "char_count": char_count,
            "avg_word_length": float(avg_word_length),
            "noun_count": noun_count,
            "verb_count": verb_count,
            "adj_count": adj_count,
            "entity_count": entity_count,
            "embedding_model": "all-MiniLM-L6-v2",
            "sentence_embedding": encoded_embedding,
            # Add product mentions
            "mentions_masterblaster": product_mentions["mentions_masterblaster"],
            "masterblaster_quantity": product_mentions["masterblaster_quantity"],
            "mentions_funpun": product_mentions["mentions_funpun"],
            "funpun_quantity": product_mentions["funpun_quantity"],
            "mentions_powerpro": product_mentions["mentions_powerpro"],
            "powerpro_quantity": product_mentions["powerpro_quantity"]
        }
        
        # Log product mentions
        if any(product_mentions[key] for key in product_mentions if key.startswith("mentions_")):
            logger.info(f"Found product mentions in sentence {request.sentence_id}: {product_mentions}")
        
        # Store features in database
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DB_SERVICE_URL}/features",
                json=feature_data,
                timeout=30.0  # Add timeout
            )
            
            if response.status_code != 201:
                logger.error(f"Failed to store features: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to store features in database"
                )
            
            logger.info(f"Features stored successfully for sentence ID: {request.sentence_id}")
            return FeatureExtractionResponse(**feature_data)
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing features: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "feature_extraction",
        "models_loaded": {
            "spacy": nlp.meta["lang"],
            "sentence_transformer": embedding_model.get_sentence_embedding_dimension() == 384
        },
        "db_service_url": DB_SERVICE_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.feature_service:app", host="0.0.0.0", port=PORT, reload=True)