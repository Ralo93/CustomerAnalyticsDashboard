from datetime import datetime
import uuid
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import enum

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/ml_insights")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Sentence(Base):
    __tablename__ = "sentences"
    
    id = Column(String, primary_key=True, index=True)
    text = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now())
    
    # Processing fields
    sentiment = Column(String, nullable=True)
    word_count = Column(Integer, nullable=True)
    processing_status = Column(String, nullable=True)
    
    # Priority fields
    priority = Column(String, default="normal", nullable=True)
    intent_type = Column(String, nullable=True)
    priority_reasoning = Column(Text, nullable=True)  # Text type for potentially longer explanations

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)