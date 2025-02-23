import base64
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
import logging
import time
from datetime import datetime
from . import models
from pydantic import BaseModel, ConfigDict, validator
from typing import List, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Database Service")

# Dependency to get DB session
def get_db():
    db = models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class BaseModelConfig(BaseModel):
    """Base Pydantic model with common configuration"""
    model_config = ConfigDict(from_attributes=True)

class SentenceBase(BaseModelConfig):
    external_id: str
    text: str

class SentenceCreate(SentenceBase):
    pass

class SentenceUpdate(BaseModelConfig):
    text: Optional[str] = None

class SentenceResponse(SentenceBase):
    id: str
    created_at: datetime

class FeatureBase(BaseModelConfig):
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    avg_word_length: Optional[float] = None
    noun_count: Optional[int] = None
    verb_count: Optional[int] = None
    adj_count: Optional[int] = None
    entity_count: Optional[int] = None
    sentiment_score: Optional[float] = None
    embedding_model: Optional[str] = None

class FeatureCreate(FeatureBase):
    sentence_id: str
    sentence_embedding: Optional[str] = None  # Base64 encoded string

    @validator('sentence_embedding')
    def validate_and_encode_embedding(cls, v):
        if v is not None:
            try:
                # If it's already base64, try to decode and verify dimension
                embedding_bytes = base64.b64decode(v)
                embedding_array = np.frombuffer(embedding_bytes, dtype=np.float64)
                if len(embedding_array) != 384:
                    raise ValueError("Embedding must be 384-dimensional")
            except Exception as e:
                raise ValueError(f"Invalid base64 encoded embedding: {str(e)}")
        return v

class FeatureUpdate(FeatureBase):
    sentence_embedding: Optional[str] = None  # Base64 encoded string

    @validator('sentence_embedding')
    def validate_and_encode_embedding(cls, v):
        if v is not None:
            try:
                # If it's already base64, try to decode and verify dimension
                embedding_bytes = base64.b64decode(v)
                embedding_array = np.frombuffer(embedding_bytes, dtype=np.float64)
                if len(embedding_array) != 384:
                    raise ValueError("Embedding must be 384-dimensional")
            except Exception as e:
                raise ValueError(f"Invalid base64 encoded embedding: {str(e)}")
        return v

class FeatureResponse(FeatureBase):
    id: str
    sentence_id: str
    created_at: datetime
    sentence_embedding: Optional[str] = None  # Will be returned as base64


class LabelBase(BaseModelConfig):
    sales_funnel_stage: Optional[str] = None
    sales_funnel_confidence: Optional[float] = None
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    business_impact: Optional[str] = None
    business_impact_confidence: Optional[float] = None

class LabelCreate(LabelBase):
    sentence_id: str

class LabelUpdate(LabelBase):
    pass


class LabelResponse(LabelBase):
    id: str
    sentence_id: str
    created_at: datetime
    last_updated: datetime




@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all requests with timing information"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
    return response

@app.on_event("startup")
async def startup_event():
    logger.info("Database Service starting up")
    models.create_tables()
    logger.info("Database tables created/verified")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Database Service shutting down")

@app.post("/sentences", status_code=201, response_model=SentenceResponse)
async def create_sentence(sentence: SentenceCreate, db: Session = Depends(get_db)):
    """Create a new sentence"""
    sentence_id = str(uuid.uuid4())
    logger.info(f"Creating new sentence with ID: {sentence_id}")
    
    db_sentence = models.Sentence(
        id=sentence_id,
        external_id=sentence.external_id,
        text=sentence.text
    )
    
    db.add(db_sentence)
    db.commit()
    db.refresh(db_sentence)
    
    logger.info(f"Sentence {sentence_id} created successfully")
    return db_sentence

@app.get("/sentences", response_model=List[SentenceResponse])
async def get_sentences(db: Session = Depends(get_db)):
    """Fetch all sentences from the database"""
    logger.info("Fetching all sentences")
    sentences = db.query(models.Sentence).all()
    logger.info(f"Retrieved {len(sentences)} sentences")
    return sentences

@app.get("/sentences/{sentence_id}", response_model=SentenceResponse)
async def get_sentence(sentence_id: str, db: Session = Depends(get_db)):
    """Get a specific sentence by ID"""
    logger.info(f"Retrieving sentence: {sentence_id}")
    sentence = db.query(models.Sentence).filter(models.Sentence.id == sentence_id).first()
    if not sentence:
        logger.warning(f"Sentence not found: {sentence_id}")
        raise HTTPException(status_code=404, detail="Sentence not found")
    return sentence

@app.patch("/sentences/{sentence_id}", response_model=SentenceResponse)
async def update_sentence(sentence_id: str, update_data: SentenceUpdate, db: Session = Depends(get_db)):
    """Update a sentence"""
    logger.info(f"Updating sentence {sentence_id}")
    sentence = db.query(models.Sentence).filter(models.Sentence.id == sentence_id).first()
    if not sentence:
        logger.warning(f"Sentence not found: {sentence_id}")
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(sentence, key, value)
    
    db.commit()
    db.refresh(sentence)
    logger.info(f"Sentence {sentence_id} updated successfully")
    return sentence
@app.post("/features", status_code=201, response_model=FeatureResponse)
async def create_features(features: FeatureCreate, db: Session = Depends(get_db)):
    """Create features for a sentence"""
    logger.info(f"Creating features for sentence: {features.sentence_id}")
    
    # Verify sentence exists
    sentence = db.query(models.Sentence).filter(models.Sentence.id == features.sentence_id).first()
    if not sentence:
        logger.warning(f"Sentence not found: {features.sentence_id}")
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    # Check if features already exist
    existing_features = db.query(models.SentenceFeatures).filter(
        models.SentenceFeatures.sentence_id == features.sentence_id
    ).first()
    if existing_features:
        logger.warning(f"Features already exist for sentence: {features.sentence_id}")
        raise HTTPException(status_code=400, detail="Features already exist for this sentence")
    
    try:
        # Convert the feature data to dict
        feature_data = features.model_dump(exclude_unset=True)
        
        # Handle the embedding if present
        if features.sentence_embedding:
            # It's already validated as base64 by the Pydantic model
            embedding_bytes = base64.b64decode(features.sentence_embedding)
            feature_data['sentence_embedding'] = embedding_bytes
        
        # Create the features
        db_features = models.SentenceFeatures(**feature_data)
        db.add(db_features)
        db.commit()
        db.refresh(db_features)
        
        # Convert binary embedding back to base64 for response
        if db_features.sentence_embedding:
            db_features.sentence_embedding = base64.b64encode(db_features.sentence_embedding).decode()
        
        logger.info(f"Features created successfully for sentence: {features.sentence_id}")
        return db_features
        
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/{sentence_id}", response_model=FeatureResponse)
async def get_features(sentence_id: str, db: Session = Depends(get_db)):
    """Get features for a sentence"""
    features = db.query(models.SentenceFeatures).filter(
        models.SentenceFeatures.sentence_id == sentence_id
    ).first()
    
    if not features:
        raise HTTPException(status_code=404, detail="Features not found")
    
    # Convert binary embedding to base64 for response if present
    if features.sentence_embedding:
        features.sentence_embedding = base64.b64encode(features.sentence_embedding).decode()
    
    return features

@app.patch("/features/{sentence_id}", response_model=FeatureResponse)
async def update_features(sentence_id: str, update_data: FeatureUpdate, db: Session = Depends(get_db)):
    """Update features for a sentence"""
    features = db.query(models.SentenceFeatures).filter(
        models.SentenceFeatures.sentence_id == sentence_id
    ).first()
    if not features:
        raise HTTPException(status_code=404, detail="Features not found")
    
    try:
        # Get the update data
        update_dict = update_data.model_dump(exclude_unset=True)
        
        # Handle the embedding if present
        if update_data.sentence_embedding:
            # It's already validated as base64 by the Pydantic model
            embedding_bytes = base64.b64decode(update_data.sentence_embedding)
            update_dict['sentence_embedding'] = embedding_bytes
        
        # Update the features
        for key, value in update_dict.items():
            setattr(features, key, value)
        
        db.commit()
        db.refresh(features)
        
        # Convert binary embedding back to base64 for response
        if features.sentence_embedding:
            features.sentence_embedding = base64.b64encode(features.sentence_embedding).decode()
        
        return features
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/labels", status_code=201, response_model=LabelResponse)
async def create_label(label: LabelCreate, db: Session = Depends(get_db)):
    """Create a label for a sentence"""
    # Verify sentence exists
    sentence = db.query(models.Sentence).filter(models.Sentence.id == label.sentence_id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    # Check if label already exists
    existing_label = db.query(models.SentenceLabel).filter(
        models.SentenceLabel.sentence_id == label.sentence_id
    ).first()
    if existing_label:
        raise HTTPException(status_code=400, detail="Label already exists for this sentence")
    
    db_label = models.SentenceLabel(**label.model_dump())
    db.add(db_label)
    db.commit()
    db.refresh(db_label)
    return db_label

@app.get("/labels/{sentence_id}", response_model=LabelResponse)
async def get_label(sentence_id: str, db: Session = Depends(get_db)):
    """Get label for a sentence"""
    label = db.query(models.SentenceLabel).filter(
        models.SentenceLabel.sentence_id == sentence_id
    ).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    return label

@app.patch("/labels/{sentence_id}", response_model=LabelResponse)
async def update_label(sentence_id: str, update_data: LabelUpdate, db: Session = Depends(get_db)):
    """Update label for a sentence"""
    label = db.query(models.SentenceLabel).filter(
        models.SentenceLabel.sentence_id == sentence_id
    ).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(label, key, value)
    
    label.last_updated = datetime.now()
    db.commit()
    db.refresh(label)
    return label

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "database_service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Database Service")
    uvicorn.run("src.db_service.service:app", host="0.0.0.0", port=8001, reload=True)