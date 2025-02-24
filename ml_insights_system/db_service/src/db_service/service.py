import base64
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import and_, case, func
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

# Response models for analytics endpoints
class SentimentDistribution(BaseModel):
    sales_funnel_stage: str
    positive_count: int
    negative_count: int
    neutral_count: int
    total_count: int

class ImpactScore(BaseModel):
    sales_funnel_stage: str
    average_impact: float
    count: int

class SentimentImpactScore(BaseModel):
    sentiment: str
    business_impact: str
    count: int

class FunnelMetrics(BaseModel):
    stage: str
    count: int
    percentage: float

class IntentDistribution(BaseModel):
    intent: str
    count: int
    percentage: float

class HighImpactSentiment(BaseModel):
    sales_funnel_stage: str
    positive_high_impact_count: int
    total_count: int
    percentage: float

class StageIntentAlignment(BaseModel):
    sales_funnel_stage: str
    intent: str
    count: int
    alignment_score: float

class ImpactInformation(BaseModel):
    information_type: str
    business_impact: str
    count: int



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


@app.get("/analytics/sentiment-distribution", response_model=List[SentimentDistribution])
async def get_sentiment_distribution(db: Session = Depends(get_db)):
    """Get sentiment distribution per sales funnel stage"""
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.count(case((models.SentenceLabel.sentiment == 'Positive', 1))).label('positive_count'),
        func.count(case((models.SentenceLabel.sentiment == 'Negative', 1))).label('negative_count'),
        func.count(case((models.SentenceLabel.sentiment == 'Neutral', 1))).label('neutral_count'),
        func.count().label('total_count')
    ).group_by(
        models.SentenceLabel.sales_funnel_stage
    ).all()
    
    return [
        SentimentDistribution(
            sales_funnel_stage=r.sales_funnel_stage,
            positive_count=r.positive_count,
            negative_count=r.negative_count,
            neutral_count=r.neutral_count,
            total_count=r.total_count
        ) for r in results
    ]
@app.get("/analytics/impact-scores", response_model=List[ImpactScore])
async def get_impact_scores(db: Session = Depends(get_db)):
    """Get average impact scores per sales funnel stage"""
    impact_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Neutral': 1.5}
    
    # Create a list of tuples for the case statement
    whens = [(models.SentenceLabel.business_impact == k, v) for k, v in impact_mapping.items()]
    
    # Create the case expression with individual arguments
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.avg(case(*whens)).label('avg_impact'),
        func.count().label('count')
    ).group_by(
        models.SentenceLabel.sales_funnel_stage
    ).all()
    
    return [
        ImpactScore(
            sales_funnel_stage=r.sales_funnel_stage,
            average_impact=float(r.avg_impact) if r.avg_impact is not None else 0.0,
            count=r.count
        ) for r in results
    ]

@app.get("/analytics/sentiment-impact", response_model=List[SentimentImpactScore])
async def get_sentiment_impact(db: Session = Depends(get_db)):
    """Get sentiment and business impact correlation"""
    results = db.query(
        models.SentenceLabel.sentiment,
        models.SentenceLabel.business_impact,
        func.count().label('count')
    ).group_by(
        models.SentenceLabel.sentiment,
        models.SentenceLabel.business_impact
    ).all()
    
    return [
        SentimentImpactScore(
            sentiment=r.sentiment,
            business_impact=r.business_impact,
            count=r.count
        ) for r in results
    ]

@app.get("/analytics/funnel-metrics", response_model=List[FunnelMetrics])
async def get_funnel_metrics(db: Session = Depends(get_db)):
    """Get volume metrics for each funnel stage"""
    total = db.query(func.count()).select_from(models.SentenceLabel).scalar()
    
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.count().label('count')
    ).group_by(
        models.SentenceLabel.sales_funnel_stage
    ).all()
    
    return [
        FunnelMetrics(
            stage=r.sales_funnel_stage,
            count=r.count,
            percentage=r.count / total * 100 if total > 0 else 0
        ) for r in results
    ]

@app.get("/analytics/intent-distribution", response_model=List[IntentDistribution])
async def get_intent_distribution(db: Session = Depends(get_db)):
    """Get distribution of communication intents"""
    total = db.query(func.count()).select_from(models.SentenceLabel).scalar()
    
    results = db.query(
        models.SentenceLabel.intent,
        func.count().label('count')
    ).group_by(
        models.SentenceLabel.intent
    ).all()
    
    return [
        IntentDistribution(
            intent=r.intent,
            count=r.count,
            percentage=r.count / total * 100 if total > 0 else 0
        ) for r in results
    ]

@app.get("/analytics/high-impact-sentiment", response_model=List[HighImpactSentiment])
async def get_high_impact_positive_sentiment(db: Session = Depends(get_db)):
    """Get high impact and positive sentiment distribution per stage"""
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.count(
            case(
                (and_(
                    models.SentenceLabel.business_impact == 'High',
                    models.SentenceLabel.sentiment == 'Positive'
                ), 1)
            )
        ).label('positive_high_impact_count'),
        func.count().label('total_count')
    ).group_by(
        models.SentenceLabel.sales_funnel_stage
    ).all()
    
    return [
        HighImpactSentiment(
            sales_funnel_stage=r.sales_funnel_stage,
            positive_high_impact_count=r.positive_high_impact_count,
            total_count=r.total_count,
            percentage=r.positive_high_impact_count / r.total_count * 100 if r.total_count > 0 else 0
        ) for r in results
    ]

@app.get("/analytics/stage-intent-alignment", response_model=List[StageIntentAlignment])
async def get_stage_intent_alignment(db: Session = Depends(get_db)):
    """Get alignment between sales funnel stages and intents"""
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        models.SentenceLabel.intent,
        func.count().label('count')
    ).group_by(
        models.SentenceLabel.sales_funnel_stage,
        models.SentenceLabel.intent
    ).all()
    
    # Calculate alignment score based on expected intent for each stage
    stage_intent_mapping = {
        'Awareness': ['Information'],
        'Interest': ['Feedback', 'Information'],
        'Consideration': ['Purchase', 'Support'],
        'Intent': ['Purchase'],
        'Evaluation': ['Support', 'Complaint'],
        'Purchase': ['Purchase', 'Feedback']
    }
    
    return [
        StageIntentAlignment(
            sales_funnel_stage=r.sales_funnel_stage,
            intent=r.intent,
            count=r.count,
            alignment_score=1.0 if r.intent in stage_intent_mapping.get(r.sales_funnel_stage, []) else 0.0
        ) for r in results
    ]

@app.get("/analytics/impact-information", response_model=List[ImpactInformation])
async def get_impact_information(db: Session = Depends(get_db)):
    """Get relationship between information type and business impact"""
    results = db.query(
        models.SentenceLabel.intent,
        models.SentenceLabel.business_impact,
        func.count().label('count')
    ).filter(
        models.SentenceLabel.intent.in_(['Information', 'Feedback', 'Support'])
    ).group_by(
        models.SentenceLabel.intent,
        models.SentenceLabel.business_impact
    ).all()
    
    return [
        ImpactInformation(
            information_type=r.intent,
            business_impact=r.business_impact,
            count=r.count
        ) for r in results
    ]


# New endpoint to fetch sentences with specific filter criteria
@app.get("/analytics/sentences-by-filter")
async def get_sentences_by_filter(
    sales_funnel_stage: Optional[str] = None,
    sentiment: Optional[str] = None,
    business_impact: Optional[str] = None,
    intent: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Fetch sentences matching specific filter criteria"""
    query = db.query(models.Sentence).join(
        models.SentenceLabel, models.Sentence.id == models.SentenceLabel.sentence_id
    )
    
    # Apply filters
    if sales_funnel_stage:
        query = query.filter(models.SentenceLabel.sales_funnel_stage == sales_funnel_stage)
    if sentiment:
        query = query.filter(models.SentenceLabel.sentiment == sentiment)
    if business_impact:
        query = query.filter(models.SentenceLabel.business_impact == business_impact)
    if intent:
        query = query.filter(models.SentenceLabel.intent == intent)
    
    # Limit results and order by newest first
    sentences = query.order_by(models.Sentence.created_at.desc()).limit(limit).all()
    
    # Return sentences with their label information
    results = []
    for sentence in sentences:
        label = db.query(models.SentenceLabel).filter(
            models.SentenceLabel.sentence_id == sentence.id
        ).first()
        
        results.append({
            "id": sentence.id,
            "text": sentence.text,
            "created_at": sentence.created_at,
            "sales_funnel_stage": label.sales_funnel_stage if label else None,
            "sentiment": label.sentiment if label else None,
            "business_impact": label.business_impact if label else None,
            "intent": label.intent if label else None
        })
    
    return results

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