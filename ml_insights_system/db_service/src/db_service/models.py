from datetime import datetime
import uuid
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, LargeBinary, String, Text, DateTime, Enum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import enum

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/ml_insights")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class LabelType(enum.Enum):
    SALES_FUNNEL = "sales_funnel"
    SENTIMENT = "sentiment"
    INTENT = "intent"
    BUSINESS_IMPACT = "business_impact"

class Sentence(Base):
    __tablename__ = "sentences"
    
    id = Column(String, primary_key=True, index=True)
    external_id = Column(String, nullable=False)
    text = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class SentenceLabel(Base):
    __tablename__ = "sentence_labels"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sentence_id = Column(String, ForeignKey('sentences.id', ondelete='CASCADE'), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Label dimensions with their confidence scores
    sales_funnel_stage = Column(String, nullable=True)
    sales_funnel_confidence = Column(Float, nullable=True)
    
    sentiment = Column(String, nullable=True)
    sentiment_confidence = Column(Float, nullable=True)
    
    intent = Column(String, nullable=True)
    intent_confidence = Column(Float, nullable=True)
    
    business_impact = Column(String, nullable=True)
    business_impact_confidence = Column(Float, nullable=True)

    is_sales_funnel_relevant = Column(Boolean, nullable=True)
    is_sales_funnel_relevant_confidence = Column(Float, nullable=True)
    
class SentenceFeatures(Base):
    __tablename__ = "sentence_features"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sentence_id = Column(String, ForeignKey('sentences.id', ondelete='CASCADE'), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Text analysis features
    word_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    avg_word_length = Column(Float, nullable=True)
    noun_count = Column(Integer, nullable=True)
    verb_count = Column(Integer, nullable=True)
    adj_count = Column(Integer, nullable=True)
    entity_count = Column(Integer, nullable=True)
    
    # Product mentions and quantities
    mentions_masterblaster = Column(Boolean, nullable=True, default=False)
    masterblaster_quantity = Column(Integer, nullable=True, default=0)
    
    mentions_funpun = Column(Boolean, nullable=True, default=False)
    funpun_quantity = Column(Integer, nullable=True, default=0)
    
    mentions_powerpro = Column(Boolean, nullable=True, default=False)
    powerpro_quantity = Column(Integer, nullable=True, default=0)
    
    # Embeddings
    sentence_embedding = Column(LargeBinary, nullable=True)  # Store as binary
    embedding_model = Column(String, nullable=True)  # Track which model generated the embedding


# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)