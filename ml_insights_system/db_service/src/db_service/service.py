import base64
import sys
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
    level=logging.DEBUG,
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
    
    # Product mention fields
    mentions_masterblaster: Optional[bool] = None
    masterblaster_quantity: Optional[int] = None
    
    mentions_funpun: Optional[bool] = None
    funpun_quantity: Optional[int] = None
    
    mentions_powerpro: Optional[bool] = None
    powerpro_quantity: Optional[int] = None

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
    is_sales_funnel_relevant: Optional[bool] = None
    is_sales_funnel_relevant_confidence: Optional[float] = None
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

@app.get("/analytics/sentiment-impact", response_model=List[SentimentImpactScore])
async def get_sentiment_impact(
    is_sales_funnel_relevant: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get sentiment and business impact correlation with optional relevance filter"""
    query = db.query(
        models.SentenceLabel.sentiment,
        models.SentenceLabel.business_impact,
        func.count().label('count')
    )
    
    # Apply relevance filter if provided
    if is_sales_funnel_relevant is not None:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    results = query.group_by(
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

@app.get("/analytics/sentiment-distribution", response_model=List[SentimentDistribution])
async def get_sentiment_distribution(db: Session = Depends(get_db)):
    """Get sentiment distribution per sales funnel stage for relevant sentences only"""
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.count(case((models.SentenceLabel.sentiment == 'Positive', 1))).label('positive_count'),
        func.count(case((models.SentenceLabel.sentiment == 'Negative', 1))).label('negative_count'),
        func.count(case((models.SentenceLabel.sentiment == 'Neutral', 1))).label('neutral_count'),
        func.count().label('total_count')
    ).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True,
        models.SentenceLabel.sales_funnel_stage.isnot(None)
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
    """Get average impact scores per sales funnel stage for relevant sentences only"""
    impact_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Neutral': 1.5}
    
    # Create a list of tuples for the case statement
    whens = [(models.SentenceLabel.business_impact == k, v) for k, v in impact_mapping.items()]
    
    # Create the case expression with individual arguments
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.avg(case(*whens)).label('avg_impact'),
        func.count().label('count')
    ).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True,
        models.SentenceLabel.sales_funnel_stage.isnot(None)
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

@app.get("/analytics/funnel-metrics", response_model=List[FunnelMetrics])
async def get_funnel_metrics(db: Session = Depends(get_db)):
    """Get volume metrics for each funnel stage for relevant sentences only"""
    # Only count funnel-relevant sentences for the total
    total = db.query(func.count()).select_from(models.SentenceLabel).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True,
        models.SentenceLabel.sales_funnel_stage.isnot(None)
    ).scalar()
    
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        func.count().label('count')
    ).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True,
        models.SentenceLabel.sales_funnel_stage.isnot(None)
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

@app.get("/analytics/high-impact-sentiment", response_model=List[HighImpactSentiment])
async def get_high_impact_positive_sentiment(db: Session = Depends(get_db)):
    """Get high impact and positive sentiment distribution per stage for relevant sentences only"""
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
    ).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True,
        models.SentenceLabel.sales_funnel_stage.isnot(None)
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

@app.get("/analytics/general-sentiment-distribution")
async def get_general_sentiment_distribution(
    is_sales_funnel_relevant: Optional[bool] = None, 
    db: Session = Depends(get_db)
):
    """Get sentiment distribution for all sentences with optional relevance filter"""
    query = db.query(
        models.SentenceLabel.sentiment,
        func.count().label('count')
    )
    
    # Apply relevance filter if provided
    if is_sales_funnel_relevant is not None:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    results = query.group_by(
        models.SentenceLabel.sentiment
    ).all()
    
    total = sum(r.count for r in results)
    
    return [
        {
            "sentiment": r.sentiment,
            "count": r.count,
            "percentage": r.count / total * 100 if total > 0 else 0
        } for r in results
    ]
@app.get("/analytics/intent-distribution", response_model=List[IntentDistribution])
async def get_intent_distribution(
    is_sales_funnel_relevant: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get distribution of communication intents with optional relevance filter"""
    query = db.query(
        models.SentenceLabel.intent,
        func.count().label('count')
    )
    
    # By default, show only non-relevant items
    if is_sales_funnel_relevant is None:
        # Default to non-relevant if parameter is not provided
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == False)
    else:
        # If parameter is explicitly provided, use that value
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    # Complete the query
    results = query.group_by(
        models.SentenceLabel.intent
    ).all()
    
    total = sum(r.count for r in results)
    
    return [
        IntentDistribution(
            intent=r.intent,
            count=r.count,
            percentage=r.count / total * 100 if total > 0 else 0
        ) for r in results
    ]

@app.get("/analytics/business-impact-distribution")
async def get_business_impact_distribution(
    is_sales_funnel_relevant: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get distribution of business impact with optional relevance filter"""
    query = db.query(
        models.SentenceLabel.business_impact,
        func.count().label('count')
    )
    
    # Apply relevance filter if provided
    if is_sales_funnel_relevant is not None:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    results = query.group_by(
        models.SentenceLabel.business_impact
    ).all()
    
    total = sum(r.count for r in results)
    
    return [
        {
            "business_impact": r.business_impact,
            "count": r.count,
            "percentage": r.count / total * 100 if total > 0 else 0
        } for r in results
    ]
@app.get("/analytics/impact-information", response_model=List[ImpactInformation])
async def get_impact_information(
    is_sales_funnel_relevant: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get relationship between information type and business impact with optional relevance filter"""
    query = db.query(
        models.SentenceLabel.intent.label('information_type'),
        models.SentenceLabel.business_impact,
        func.count().label('count')
    ).filter(
        models.SentenceLabel.intent.in_(['Information', 'Feedback', 'Support'])
    )
    
    # Apply relevance filter if provided
    if is_sales_funnel_relevant is not None:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    results = query.group_by(
        models.SentenceLabel.intent,
        models.SentenceLabel.business_impact
    ).all()
    
    return [
        ImpactInformation(
            information_type=r.information_type,
            business_impact=r.business_impact,
            count=r.count
        ) for r in results
    ]

@app.get("/analytics/stage-intent-alignment", response_model=List[StageIntentAlignment])
async def get_stage_intent_alignment(db: Session = Depends(get_db)):
    """Get alignment between sales funnel stages and intents for relevant sentences only"""
    results = db.query(
        models.SentenceLabel.sales_funnel_stage,
        models.SentenceLabel.intent,
        func.count().label('count')
    ).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True,
        models.SentenceLabel.sales_funnel_stage.isnot(None)
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

@app.get("/analytics/sentences-by-filter")
async def get_sentences_by_filter(
    sales_funnel_stage: Optional[str] = None,
    sentiment: Optional[str] = None,
    business_impact: Optional[str] = None,
    intent: Optional[str] = None,
    is_sales_funnel_relevant: Optional[bool] = None,
    limit: int = sys.maxsize,
    db: Session = Depends(get_db)
):
    """Fetch sentences matching specific filter criteria"""
    query = db.query(models.Sentence).join(
        models.SentenceLabel, models.Sentence.id == models.SentenceLabel.sentence_id
    )
    
    # Apply filters
    if is_sales_funnel_relevant is not None:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    # Only filter by sales_funnel_stage if the sentence is relevant to the sales funnel
    if sales_funnel_stage:
        query = query.filter(
            models.SentenceLabel.is_sales_funnel_relevant == True,
            models.SentenceLabel.sales_funnel_stage == sales_funnel_stage
        )
    
    # Apply other filters
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
            "is_sales_funnel_relevant": label.is_sales_funnel_relevant if label else None,
            "sales_funnel_stage": label.sales_funnel_stage if label and label.is_sales_funnel_relevant else None,
            "sentiment": label.sentiment if label else None,
            "business_impact": label.business_impact if label else None,
            "intent": label.intent if label else None
        })
    
    return results

@app.get("/analytics/all-sentences")
async def get_all_sentences(
    include_non_relevant: bool = True,
    limit: int = 1000, 
    db: Session = Depends(get_db)
):
    """Fetch all sentences with their label data for client-side filtering
    
    Args:
        include_non_relevant: Whether to include sentences not relevant to sales funnel
        limit: Maximum number of sentences to return (default: 1000)
        db: Database session
        
    Returns:
        List of sentence objects with their label data
    """
    # Start query
    query = db.query(models.Sentence).join(
        models.SentenceLabel, models.Sentence.id == models.SentenceLabel.sentence_id
    ).outerjoin(  # Use outer join in case some sentences don't have features
        models.SentenceFeatures, models.Sentence.id == models.SentenceFeatures.sentence_id
    )
    
    # Apply relevance filter if requested
    if not include_non_relevant:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == True)
    
    # Finish query
    sentences = query.order_by(models.Sentence.created_at.desc()).limit(limit).all()
    
    # Build response with sentence and label data combined
    results = []
    for sentence in sentences:
        label = db.query(models.SentenceLabel).filter(
            models.SentenceLabel.sentence_id == sentence.id
        ).first()

                # Get feature data for this sentence including product mentions
        features = db.query(models.SentenceFeatures).filter(
            models.SentenceFeatures.sentence_id == sentence.id
        ).first()
        
        if label:
            # Only include sales_funnel_stage if the sentence is relevant
            sales_funnel_stage = label.sales_funnel_stage if label.is_sales_funnel_relevant else None
            sales_funnel_confidence = label.sales_funnel_confidence if label.is_sales_funnel_relevant else None
            
            results.append({
                "id": sentence.id,
                "text": sentence.text,
                "created_at": sentence.created_at.isoformat(),
                "is_sales_funnel_relevant": label.is_sales_funnel_relevant,
                "is_sales_funnel_relevant_confidence": label.is_sales_funnel_relevant_confidence,
                "sales_funnel_stage": sales_funnel_stage,
                "sales_funnel_confidence": sales_funnel_confidence,
                "sentiment": label.sentiment,
                "sentiment_confidence": label.sentiment_confidence,
                "business_impact": label.business_impact,
                "business_impact_confidence": label.business_impact_confidence,
                "intent": label.intent,
                "intent_confidence": label.intent_confidence,
                "mentions_masterblaster": features.mentions_masterblaster if features else False,
                "mentions_funpun": features.mentions_funpun if features else False,
                "mentions_powerpro": features.mentions_powerpro if features else False,
            })
    
    return results

# New endpoint to get sales funnel relevance statistics
@app.get("/analytics/sales-funnel-relevance")
async def get_sales_funnel_relevance(db: Session = Depends(get_db)):
    """Get statistics about sales funnel relevance of sentences"""
    
    # Get total count
    total = db.query(func.count()).select_from(models.SentenceLabel).scalar()
    
    # Get relevant count
    relevant = db.query(func.count()).select_from(models.SentenceLabel).filter(
        models.SentenceLabel.is_sales_funnel_relevant == True
    ).scalar()
    
    # Get non-relevant count
    non_relevant = db.query(func.count()).select_from(models.SentenceLabel).filter(
        models.SentenceLabel.is_sales_funnel_relevant == False
    ).scalar()
    
    # Get count without classification
    unclassified = db.query(func.count()).select_from(models.SentenceLabel).filter(
        models.SentenceLabel.is_sales_funnel_relevant.is_(None)
    ).scalar()
    
    return {
        "total_sentences": total,
        "sales_funnel_relevant": relevant,
        "non_sales_funnel_relevant": non_relevant,
        "unclassified": unclassified,
        "relevant_percentage": (relevant / total * 100) if total > 0 else 0,
        "non_relevant_percentage": (non_relevant / total * 100) if total > 0 else 0,
        "unclassified_percentage": (unclassified / total * 100) if total > 0 else 0
    }

@app.get("/analytics/product-mentions")
async def get_product_mentions(
    is_sales_funnel_relevant: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    
    # First, log raw data for diagnostic purposes
    raw_features = db.query(models.SentenceFeatures).all()
    
    logger.debug("Raw SentenceFeatures data:")
    for feature in raw_features:
        logger.debug(f"Sentence ID: {feature.sentence_id}")
        logger.debug(f"MasterBlaster Mentions: {feature.mentions_masterblaster}, Quantity: {feature.masterblaster_quantity}")
        logger.debug(f"FunPun Mentions: {feature.mentions_funpun}, Quantity: {feature.funpun_quantity}")
        logger.debug(f"PowerPro Mentions: {feature.mentions_powerpro}, Quantity: {feature.powerpro_quantity}")
    
    """Get lightweight statistics about product mentions in sentences"""
    query = db.query(
        func.count(models.Sentence.id).label('total_sentences'),
        func.sum(case(
            (models.SentenceFeatures.mentions_masterblaster == True, 1), 
            else_=0
        )).label('masterblaster_mentions'),
        func.sum(case(
            (models.SentenceFeatures.mentions_funpun == True, 1), 
            else_=0
        )).label('funpun_mentions'),
        func.sum(case(
            (models.SentenceFeatures.mentions_powerpro == True, 1), 
            else_=0
        )).label('powerpro_mentions'),
        func.sum(case(
            ((models.SentenceFeatures.mentions_masterblaster == True) | 
             (models.SentenceFeatures.mentions_funpun == True) | 
             (models.SentenceFeatures.mentions_powerpro == True), 1),
            else_=0
        )).label('sentences_with_products'),
        func.sum(case(
            (((models.SentenceFeatures.mentions_masterblaster == True) and 
              (models.SentenceFeatures.mentions_funpun == True or models.SentenceFeatures.mentions_powerpro == True)) or
             ((models.SentenceFeatures.mentions_funpun == True) and 
              (models.SentenceFeatures.mentions_masterblaster == True or models.SentenceFeatures.mentions_powerpro == True)) or
             ((models.SentenceFeatures.mentions_powerpro == True) and 
              (models.SentenceFeatures.mentions_masterblaster == True or models.SentenceFeatures.mentions_funpun == True)),
            1),
            else_=0
        )).label('multiple_product_mentions'),
        # Sum the quantity fields, converting nulls to 0 using func.coalesce
        func.sum(case(
            (models.SentenceFeatures.mentions_masterblaster == True,
             func.coalesce(models.SentenceFeatures.masterblaster_quantity, 0)),
            else_=0
        )).label('masterblaster_quantity'),
        func.sum(case(
            (models.SentenceFeatures.mentions_funpun == True,
             func.coalesce(models.SentenceFeatures.funpun_quantity, 0)),
            else_=0
        )).label('funpun_quantity'),
        func.sum(case(
            (models.SentenceFeatures.mentions_powerpro == True,
             func.coalesce(models.SentenceFeatures.powerpro_quantity, 0)),
            else_=0
        )).label('powerpro_quantity'),
    ).join(
        models.SentenceFeatures, 
        models.Sentence.id == models.SentenceFeatures.sentence_id
    )
    
    if is_sales_funnel_relevant is not None:
        query = query.join(
            models.SentenceLabel,
            models.Sentence.id == models.SentenceLabel.sentence_id
        ).filter(
            models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant
        )
    
    result = query.first()

        # Log the raw query result for debugging
    logger.debug(f"Raw query result: {result}")
    
    if not result:
        return {
            "total_sentences": 0,
            "with_any_product_mention": 0,
            "with_multiple_products": 0,
            "masterblaster": {"mention_count": 0, "quantity": 0, "percentage": 0},
            "funpun": {"mention_count": 0, "quantity": 0, "percentage": 0},
            "powerpro": {"mention_count": 0, "quantity": 0, "percentage": 0}
        }
    
    total_sentences = result.total_sentences or 0
    
    product_stats = {
        "total_sentences": total_sentences,
        "with_any_product_mention": result.sentences_with_products or 0,
        "with_multiple_products": result.multiple_product_mentions or 0,
        "masterblaster": {
            "mention_count": result.masterblaster_mentions or 0,
            "quantity": result.masterblaster_quantity or 0,
            "percentage": (result.masterblaster_mentions / total_sentences * 100) if total_sentences > 0 else 0
        },
        "funpun": {
            "mention_count": result.funpun_mentions or 0,
            "quantity": result.funpun_quantity or 0,
            "percentage": (result.funpun_mentions / total_sentences * 100) if total_sentences > 0 else 0
        },
        "powerpro": {
            "mention_count": result.powerpro_mentions or 0,
            "quantity": result.powerpro_quantity or 0,
            "percentage": (result.powerpro_mentions / total_sentences * 100) if total_sentences > 0 else 0
        }
    }
    
    return product_stats


@app.get("/analytics/sentences-by-product-filter")
async def get_sentences_by_product_filter(
    mentions_masterblaster: Optional[bool] = None,
    mentions_funpun: Optional[bool] = None,
    mentions_powerpro: Optional[bool] = None,
    min_masterblaster_quantity: Optional[int] = None,
    min_funpun_quantity: Optional[int] = None,
    min_powerpro_quantity: Optional[int] = None,
    is_sales_funnel_relevant: Optional[bool] = None,
    sentiment: Optional[str] = None,
    business_impact: Optional[str] = None,
    intent: Optional[str] = None,
    limit: int = sys.maxsize,
    db: Session = Depends(get_db)
):
    """Fetch sentences matching product mention criteria"""
    # Start with a query that joins all necessary tables
    query = db.query(models.Sentence).join(
        models.SentenceFeatures, models.Sentence.id == models.SentenceFeatures.sentence_id
    ).join(
        models.SentenceLabel, models.Sentence.id == models.SentenceLabel.sentence_id
    )
    
    # Apply product mention filters
    if mentions_masterblaster is not None:
        query = query.filter(models.SentenceFeatures.mentions_masterblaster == mentions_masterblaster)
    
    if mentions_funpun is not None:
        query = query.filter(models.SentenceFeatures.mentions_funpun == mentions_funpun)
    
    if mentions_powerpro is not None:
        query = query.filter(models.SentenceFeatures.mentions_powerpro == mentions_powerpro)
    
    # Apply product quantity filters
    if min_masterblaster_quantity is not None and min_masterblaster_quantity > 0:
        query = query.filter(models.SentenceFeatures.masterblaster_quantity >= min_masterblaster_quantity)
    
    if min_funpun_quantity is not None and min_funpun_quantity > 0:
        query = query.filter(models.SentenceFeatures.funpun_quantity >= min_funpun_quantity)
    
    if min_powerpro_quantity is not None and min_powerpro_quantity > 0:
        query = query.filter(models.SentenceFeatures.powerpro_quantity >= min_powerpro_quantity)
    
    # Apply other filters
    if is_sales_funnel_relevant is not None:
        query = query.filter(models.SentenceLabel.is_sales_funnel_relevant == is_sales_funnel_relevant)
    
    if sentiment:
        query = query.filter(models.SentenceLabel.sentiment == sentiment)
    
    if business_impact:
        query = query.filter(models.SentenceLabel.business_impact == business_impact)
    
    if intent:
        query = query.filter(models.SentenceLabel.intent == intent)
    
    # Limit results and order by newest first
    sentences = query.order_by(models.Sentence.created_at.desc()).limit(limit).all()
    
    # Return sentences with their label and feature information
    results = []
    for sentence in sentences:
        label = db.query(models.SentenceLabel).filter(
            models.SentenceLabel.sentence_id == sentence.id
        ).first()
        
        features = db.query(models.SentenceFeatures).filter(
            models.SentenceFeatures.sentence_id == sentence.id
        ).first()
        
        results.append({
            "id": sentence.id,
            "text": sentence.text,
            "created_at": sentence.created_at.isoformat(),
            "is_sales_funnel_relevant": label.is_sales_funnel_relevant if label else None,
            "sales_funnel_stage": label.sales_funnel_stage if label and label.is_sales_funnel_relevant else None,
            "sentiment": label.sentiment if label else None,
            "business_impact": label.business_impact if label else None,
            "intent": label.intent if label else None,
            "mentions_masterblaster": features.mentions_masterblaster if features else False,
            "mentions_funpun": features.mentions_funpun if features else False,
            "mentions_powerpro": features.mentions_powerpro if features else False,
            "masterblaster_quantity": features.masterblaster_quantity if features and features.mentions_masterblaster else None,
            "funpun_quantity": features.funpun_quantity if features and features.mentions_funpun else None,
            "powerpro_quantity": features.powerpro_quantity if features and features.mentions_powerpro else None
        })
    
    return results

@app.get("/analytics/time-series-trends")
async def get_time_series_trends(
    start_external_id: Optional[int] = None,
    end_external_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get trend data across a range of external_ids (proxy for time series)"""
    # Initial query to get all sentences with their labels and features
    base_query = db.query(
        models.Sentence.external_id,
        models.SentenceLabel.sentiment,
        models.SentenceLabel.business_impact,
        models.SentenceLabel.is_sales_funnel_relevant,
        models.SentenceLabel.sales_funnel_stage,
        models.SentenceFeatures.mentions_masterblaster,
        models.SentenceFeatures.mentions_funpun,
        models.SentenceFeatures.mentions_powerpro
    ).join(
        models.SentenceLabel, 
        models.Sentence.id == models.SentenceLabel.sentence_id
    ).outerjoin(
        models.SentenceFeatures,
        models.Sentence.id == models.SentenceFeatures.sentence_id
    )
    
    # Get all sentences and filter in Python (more reliable than SQL for mixed string/int comparisons)
    all_results = base_query.all()
    
    # Filter and sort numerically
    filtered_results = []
    for result in all_results:
        try:
            # Convert external_id to integer
            ext_id = int(result.external_id)
            
            # Apply range filters if provided
            if start_external_id is not None and ext_id < start_external_id:
                continue
            if end_external_id is not None and ext_id > end_external_id:
                continue
                
            filtered_results.append((ext_id, result))
        except (ValueError, TypeError):
            # Skip results with non-numeric external_ids
            logger.warning(f"Skipping non-numeric external_id: {result.external_id}")
            continue
    
    # Sort numerically by the integer value of external_id
    sorted_results = sorted(filtered_results, key=lambda pair: pair[0])
    
    if not sorted_results:
        return {
            "time_points": [],
            "sentiment_trends": [],
            "product_trends": [],
            "impact_trends": [],
            "funnel_stage_trends": []
        }
    
    # No bucketing - use every data point in the filtered range
    time_points = [pair[0] for pair in sorted_results]
    individual_results = [pair[1] for pair in sorted_results]
    
    # Process each data point individually
    sentiment_trends = []
    product_trends = []
    impact_trends = []
    funnel_stage_trends = []
    
    for result in individual_results:
        # Calculate sentiment data for this point
        sentiment_trends.append({
            'positive': 1 if result.sentiment == 'Positive' else 0,
            'negative': 1 if result.sentiment == 'Negative' else 0,
            'neutral': 1 if result.sentiment == 'Neutral' else 0,
            'total': 1
        })
        
        # Calculate product mention data for this point
        product_trends.append({
            'masterblaster': 1 if result.mentions_masterblaster else 0,
            'funpun': 1 if result.mentions_funpun else 0,
            'powerpro': 1 if result.mentions_powerpro else 0,
            'any_product': 1 if (result.mentions_masterblaster or result.mentions_funpun or result.mentions_powerpro) else 0,
            'total': 1
        })
        
        # Calculate business impact data for this point
        impact_trends.append({
            'high': 1 if result.business_impact == 'High' else 0,
            'medium': 1 if result.business_impact == 'Medium' else 0,
            'low': 1 if result.business_impact == 'Low' else 0,
            'neutral': 1 if result.business_impact == 'Neutral' else 0,
            'total': 1
        })
        
        # Calculate sales funnel stage data for this point (only if relevant)
        # For non-relevant sentences, all stage counts are 0
        is_relevant = result.is_sales_funnel_relevant
        funnel_stage_trends.append({
            'awareness': 1 if (is_relevant and result.sales_funnel_stage == 'Awareness') else 0,
            'interest': 1 if (is_relevant and result.sales_funnel_stage == 'Interest') else 0,
            'consideration': 1 if (is_relevant and result.sales_funnel_stage == 'Consideration') else 0,
            'intent': 1 if (is_relevant and result.sales_funnel_stage == 'Intent') else 0,
            'evaluation': 1 if (is_relevant and result.sales_funnel_stage == 'Evaluation') else 0,
            'purchase': 1 if (is_relevant and result.sales_funnel_stage == 'Purchase') else 0,
            'total': 1 if is_relevant else 0
        })
    
    # Return all trend data
    return {
        "time_points": time_points,
        "sentiment_trends": sentiment_trends,
        "product_trends": product_trends,
        "impact_trends": impact_trends,
        "funnel_stage_trends": funnel_stage_trends
    }

@app.get("/analytics/external-id-range")
async def get_external_id_range(db: Session = Depends(get_db)):
    """Get the range of external_ids available in the database"""
    # Get all external IDs
    external_ids = db.query(models.Sentence.external_id).all()
    
    # Extract and convert to integers where possible
    numeric_ids = []
    for id_tuple in external_ids:
        try:
            numeric_ids.append(int(id_tuple[0]))
        except (ValueError, TypeError):
            # Skip IDs that can't be converted
            logger.warning(f"Skipping non-numeric external_id: {id_tuple[0]}")
    
    if not numeric_ids:
        return {
            "min_id": 0,
            "max_id": 0,
            "count": 0
        }
    
    # Sort numerically and get min/max
    numeric_ids.sort()
    min_id = numeric_ids[0]
    max_id = numeric_ids[-1]
    
    # Get total count
    count = len(numeric_ids)
    
    return {
        "min_id": min_id,
        "max_id": max_id,
        "count": count
    }

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