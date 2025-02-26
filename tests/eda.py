#!/usr/bin/env python
"""
EDA and Visualization Script for ML Insights Database

This script connects to the database, extracts data from:
  - sentences
  - sentence_labels
  - sentence_features

It then creates several interactive visualizations using Plotly:
  • Sentiment distribution by sales funnel stage (grouped bar chart)
  • Product mentions count (bar chart)
  • Distribution of word counts (histogram)
  • Time series trend of sentence creation (line chart)
  • (Optional) Business impact and intent distributions

Before running, ensure the DATABASE_URL environment variable is set
or update the default connection string below.
"""

import os
import uuid
from datetime import datetime
import pandas as pd
import plotly.express as px

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# -------------------------
# Database and ORM Setup
# -------------------------

# Use your DATABASE_URL environment variable or default to local Postgres connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/ml_insights")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define the models (only including fields needed for EDA)
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
    
    # Label dimensions
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


# -------------------------
# Data Extraction Function
# -------------------------

def load_data():
    """
    Query the database to join Sentence, SentenceLabel, and SentenceFeatures.
    Returns a pandas DataFrame.
    """
    session = SessionLocal()
    try:
        # Join sentences with labels and features (features may be missing, so use outer join)
        records = []
        query = (
            session.query(Sentence, SentenceLabel, SentenceFeatures)
            .join(SentenceLabel, Sentence.id == SentenceLabel.sentence_id)
            .outerjoin(SentenceFeatures, Sentence.id == SentenceFeatures.sentence_id)
        )
        for sentence, label, features in query.all():
            record = {
                "sentence_id": sentence.id,
                "external_id": sentence.external_id,
                "text": sentence.text,
                "created_at": sentence.created_at,
                "is_sales_funnel_relevant": label.is_sales_funnel_relevant if label else None,
                "sales_funnel_stage": label.sales_funnel_stage if label else None,
                "sentiment": label.sentiment if label else None,
                "sentiment_confidence": label.sentiment_confidence if label else None,
                "business_impact": label.business_impact if label else None,
                "business_impact_confidence": label.business_impact_confidence if label else None,
                "intent": label.intent if label else None,
                "intent_confidence": label.intent_confidence if label else None,
                "word_count": features.word_count if features else None,
                "char_count": features.char_count if features else None,
                "avg_word_length": features.avg_word_length if features else None,
                "noun_count": features.noun_count if features else None,
                "verb_count": features.verb_count if features else None,
                "adj_count": features.adj_count if features else None,
                "entity_count": features.entity_count if features else None,
                "mentions_masterblaster": features.mentions_masterblaster if features else False,
                "masterblaster_quantity": features.masterblaster_quantity if features else 0,
                "mentions_funpun": features.mentions_funpun if features else False,
                "funpun_quantity": features.funpun_quantity if features else 0,
                "mentions_powerpro": features.mentions_powerpro if features else False,
                "powerpro_quantity": features.powerpro_quantity if features else 0,
            }
            records.append(record)
        df = pd.DataFrame(records)
        return df
    finally:
        session.close()


# -------------------------
# Visualization Functions
# -------------------------

def plot_sentiment_by_funnel(df):
    """Bar chart: Sentiment distribution grouped by sales funnel stage."""
    # Filter for rows with a defined sales funnel stage
    df_stage = df[df["sales_funnel_stage"].notnull()]
    # Group and count sentiments per funnel stage
    sentiment_counts = (
        df_stage.groupby(["sales_funnel_stage", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        sentiment_counts,
        x="sales_funnel_stage",
        y="count",
        color="sentiment",
        barmode="group",
        title="Sentiment Distribution by Sales Funnel Stage"
    )
    fig.show()


def plot_product_mentions(df):
    """Bar chart: Count of product mentions for each product."""
    product_counts = {
        "MasterBlaster": df["mentions_masterblaster"].sum(),
        "FunPun": df["mentions_funpun"].sum(),
        "PowerPro": df["mentions_powerpro"].sum()
    }
    prod_df = pd.DataFrame(list(product_counts.items()), columns=["Product", "Mentions"])
    fig = px.bar(
        prod_df,
        x="Product",
        y="Mentions",
        title="Product Mentions Count"
    )
    fig.show()


def plot_word_count_distribution(df):
    """Histogram: Distribution of word counts in sentences."""
    # Drop any missing values
    fig = px.histogram(
        df.dropna(subset=["word_count"]),
        x="word_count",
        nbins=30,
        title="Distribution of Word Count in Sentences"
    )
    fig.show()


def plot_time_series_sentences(df):
    """Line chart: Number of sentences created over time."""
    # Convert created_at to datetime.date
    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    date_counts = df.groupby("date").size().reset_index(name="count")
    fig = px.line(
        date_counts,
        x="date",
        y="count",
        title="Number of Sentences over Time"
    )
    fig.show()


def plot_business_impact_distribution(df):
    """Pie chart: Distribution of business impact values."""
    impact_counts = df["business_impact"].value_counts().reset_index()
    impact_counts.columns = ["Business Impact", "Count"]
    fig = px.pie(
        impact_counts,
        names="Business Impact",
        values="Count",
        title="Business Impact Distribution"
    )
    fig.show()


def plot_intent_distribution(df):
    """Bar chart: Distribution of communication intents."""
    intent_counts = df["intent"].value_counts().reset_index()
    intent_counts.columns = ["Intent", "Count"]
    fig = px.bar(
        intent_counts,
        x="Intent",
        y="Count",
        title="Intent Distribution"
    )
    fig.show()


# -------------------------
# Main EDA Routine
# -------------------------

def main():
    print("Loading data from the database...")
    df = load_data()
    print("Data loaded. Number of records:", len(df))
    
    # Preview the DataFrame
    print(df.head())
    
    # Create visualizations
    print("Plotting sentiment distribution by sales funnel stage...")
    plot_sentiment_by_funnel(df)
    
    print("Plotting product mentions...")
    plot_product_mentions(df)
    
    print("Plotting word count distribution...")
    plot_word_count_distribution(df)
    
    #print("Plotting time series of sentence creation...")
    #plot_time_series_sentences(df)
    
    print("Plotting business impact distribution...")
    plot_business_impact_distribution(df)
    
    print("Plotting intent distribution...")
    plot_intent_distribution(df)


if __name__ == "__main__":
    main()
