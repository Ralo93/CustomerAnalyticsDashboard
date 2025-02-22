from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import uuid
import logging
import time
from datetime import datetime
from . import models
from pydantic import BaseModel
from typing import List, Optional, Union
from contextlib import asynccontextmanager

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

class SentenceBase(BaseModel):
    text: str
    priority: Optional[str] = None
    intent_type: Optional[str] = None
    sentiment: Optional[str] = None
    word_count: Optional[int] = None
    processing_status: Optional[str] = None
    priority_reasoning: Optional[str] = None

class SentenceCreate(SentenceBase):
    pass

class SentenceUpdate(BaseModel):
    text: Optional[str] = None
    priority: Optional[str] = None
    intent_type: Optional[str] = None
    sentiment: Optional[str] = None
    word_count: Optional[int] = None
    processing_status: Optional[str] = None
    priority_reasoning: Optional[str] = None
    
    class Config:
        from_attributes = True

class SentenceResponse(SentenceBase):
    id: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all requests with timing information"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Use a short UUID for request tracking
    
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


@app.patch("/sentences/{sentence_id}")
async def update_sentence(
    sentence_id: str, 
    update_data: SentenceUpdate,
    db: Session = Depends(get_db)
):
    """Update a sentence with processing results"""
    logger.info(f"Updating sentence {sentence_id}")
    
    # Find the sentence
    sentence = db.query(models.Sentence).filter(models.Sentence.id == sentence_id).first()
    
    if not sentence:
        logger.warning(f"Sentence not found: {sentence_id}")
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    # Update the sentence with the provided data
    # Use model_dump with exclude_unset for newer Pydantic versions
    try:
        # For Pydantic v2
        update_dict = update_data.model_dump(exclude_unset=True)
        logger.debug(f"Using Pydantic v2 model_dump")
    except AttributeError:
        # Fallback for Pydantic v1
        update_dict = update_data.dict(exclude_unset=True)
        logger.debug(f"Using Pydantic v1 dict")
    
    for key, value in update_dict.items():
        if hasattr(sentence, key):
            setattr(sentence, key, value)
    
    # Commit the changes
    db.commit()
    db.refresh(sentence)
    
    logger.info(f"Sentence {sentence_id} updated successfully")
    return sentence


@app.post("/sentences", status_code=201, response_model=SentenceResponse)
async def create_sentence(sentence: SentenceCreate, db: Session = Depends(get_db)):
    """Create a new sentence"""
    # Generate a unique ID
    sentence_id = str(uuid.uuid4())
    logger.info(f"Creating new sentence with ID: {sentence_id}")
    logger.debug(f"Sentence text: {sentence.text[:50]}{'...' if len(sentence.text) > 50 else ''}")
    
    # Handle priority field conversion from string to enum if needed
    priority = sentence.priority
    
    # Create the sentence model
    db_sentence = models.Sentence(
        id=sentence_id,
        text=sentence.text,
        priority=priority,
        intent_type=sentence.intent_type,
        sentiment=sentence.sentiment,
        word_count=sentence.word_count,
        processing_status=sentence.processing_status,
        priority_reasoning=sentence.priority_reasoning
    )
    
    # Add to database
    db.add(db_sentence)
    db.commit()
    db.refresh(db_sentence)
    
    logger.info(f"Sentence {sentence_id} created successfully")
    return db_sentence

@app.get("/sentences", response_model=List[SentenceResponse])
async def get_sentences(db: Session = Depends(get_db)):
    """Fetch all sentences from the database"""
    logger.info("Fetching all sentences from the database")
    
    # Query all sentences from the database
    sentences = db.query(models.Sentence).all()
    
    logger.info(f"Retrieved {len(sentences)} sentences")
    return sentences

@app.get("/sentences/{sentence_id}")
async def get_sentence(sentence_id: str, db: Session = Depends(get_db)):
    logger.info(f"Retrieving sentence: {sentence_id}")
    
    sentence = db.query(models.Sentence).filter(models.Sentence.id == sentence_id).first()
    if not sentence:
        logger.warning(f"Sentence not found: {sentence_id}")
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    logger.info(f"Retrieved sentence {sentence_id} successfully")
    return sentence


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "database_service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Database Service")
    uvicorn.run("src.db_service.service:app", host="0.0.0.0", port=8001, reload=True)