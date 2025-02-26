#!/usr/bin/env python
"""
EDA and Visualization Script for ML Insights Database
"""

import os
import uuid
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
# Helper Functions
# -------------------------

def save_fig(fig, filename):
    """Save figure to HTML and optionally to image formats."""
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    # Save as interactive HTML
    html_path = f"figures/{filename}.html"
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"Saved interactive figure to {html_path}")


# -------------------------
# Visualization Functions
# -------------------------

def plot_sentiment_by_funnel(df):
    """Bar chart: Sentiment distribution grouped by sales funnel stage."""
    # Filter for rows with a defined sales funnel stage
    df_stage = df[df["sales_funnel_stage"].notna()]
    
    # Group and count sentiments per funnel stage
    sentiment_counts = (
        df_stage.groupby(["sales_funnel_stage", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    
    # Create the figure with simple styling
    fig = px.bar(
        sentiment_counts,
        x="sales_funnel_stage",
        y="count",
        color="sentiment",
        barmode="group",
        title="Sentiment by Sales Funnel Stage",
        labels={
            "sales_funnel_stage": "Sales Funnel Stage",
            "count": "Count",
            "sentiment": "Sentiment"
        },
        color_discrete_map={
            "Positive": "green",
            "Neutral": "gray",
            "Negative": "red"
        }
    )
    
    # Add simple formatting with larger fonts
    fig.update_layout(
        title=dict(
            text="Sentiment by Sales Funnel Stage",
            font=dict(size=24)
        ),
        xaxis=dict(
            title="Sales Funnel Stage",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        legend=dict(
            title="Sentiment",
            font=dict(size=16)
        ),
        height=600,
        width=900
    )
    
    # Save and show
    save_fig(fig, "sentiment_by_funnel")
    return fig


def plot_product_mentions(df):
    """Simple bar chart: Count of product mentions for each product."""
    product_counts = {
        "MasterBlaster": df["mentions_masterblaster"].sum(),
        "FunPun": df["mentions_funpun"].sum(),
        "PowerPro": df["mentions_powerpro"].sum()
    }
    prod_df = pd.DataFrame(list(product_counts.items()), columns=["Product", "Mentions"])
    
    # Create the figure with simple styling
    fig = px.bar(
        prod_df,
        x="Product",
        y="Mentions",
        title="Product Mentions",
        text="Mentions"  # Show values on bars
    )
    
    # Add simple formatting with larger fonts
    fig.update_layout(
        title=dict(
            text="Product Mentions",
            font=dict(size=24)
        ),
        xaxis=dict(
            title="Product",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        height=500,
        width=800
    )
    
    # Position text above bars
    fig.update_traces(textposition="outside")
    
    # Save and show
    save_fig(fig, "product_mentions")
    return fig


def plot_word_count_distribution(df):
    """Histogram: Distribution of word counts in sentences."""
    # Drop any missing values
    df_word = df.dropna(subset=["word_count"])
    
    # Create the figure with simple styling
    fig = px.histogram(
        df_word,
        x="word_count",
        nbins=20,
        title="Word Count Distribution",
        labels={"word_count": "Words per Sentence", "count": "Frequency"}
    )
    
    # Add simple formatting with larger fonts
    fig.update_layout(
        title=dict(
            text="Word Count Distribution",
            font=dict(size=24)
        ),
        xaxis=dict(
            title="Words per Sentence",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Frequency",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        height=500,
        width=800
    )
    
    # Save and show
    save_fig(fig, "word_count_distribution")
    return fig


def plot_business_impact_distribution(df):
    """Bar chart: Distribution of business impact values (avoiding pie chart issues)."""
    # Skip null values
    df_impact = df[df["business_impact"].notna()]
    
    # Count occurrences of each impact level
    impact_counts = df_impact["business_impact"].value_counts().reset_index()
    impact_counts.columns = ["Business Impact", "Count"]
    
    # Use a simple bar chart instead of a pie chart
    fig = px.bar(
        impact_counts,
        x="Business Impact",
        y="Count",
        title="Business Impact Distribution",
        text="Count"  # Show values on bars
    )
    
    # Add simple formatting with larger fonts
    fig.update_layout(
        title=dict(
            text="Business Impact Distribution",
            font=dict(size=24)
        ),
        xaxis=dict(
            title="Business Impact",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        height=500,
        width=800
    )
    
    # Position text above bars
    fig.update_traces(textposition="outside")
    
    # Save and show
    save_fig(fig, "business_impact_distribution")
    return fig


def plot_sales_funnel_classification(df):
    """Bar chart: Distribution of sales funnel stages."""
    # Filter for rows with a defined sales funnel stage
    df_funnel = df[df["sales_funnel_stage"].notna()]
    
    # Count by funnel stage
    funnel_counts = df_funnel["sales_funnel_stage"].value_counts().reset_index()
    funnel_counts.columns = ["Stage", "Count"]
    
    # Create the figure with simple styling (as a bar chart for simplicity)
    fig = px.bar(
        funnel_counts,
        x="Stage",
        y="Count",
        title="Sales Funnel Classification",
        text="Count"  # Show values on bars
    )
    
    # Add simple formatting with larger fonts
    fig.update_layout(
        title=dict(
            text="Sales Funnel Classification",
            font=dict(size=24)
        ),
        xaxis=dict(
            title="Sales Funnel Stage",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        height=500,
        width=800
    )
    
    # Position text above bars
    fig.update_traces(textposition="outside")
    
    # Save and show
    save_fig(fig, "sales_funnel_classification")
    return fig


def plot_intent_distribution(df):
    """Bar chart: Distribution of communication intents."""
    # Skip null values
    df_intent = df[df["intent"].notna()]
    
    # Count occurrences of each intent
    intent_counts = df_intent["intent"].value_counts().reset_index()
    intent_counts.columns = ["Intent", "Count"]
    
    # Create a simple bar chart
    fig = px.bar(
        intent_counts,
        x="Intent",
        y="Count",
        title="Intent Distribution",
        text="Count"  # Show values on bars
    )
    
    # Add simple formatting with larger fonts
    fig.update_layout(
        title=dict(
            text="Intent Distribution",
            font=dict(size=24)
        ),
        xaxis=dict(
            title="Intent",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        height=500,
        width=800
    )
    
    # Position text above bars
    fig.update_traces(textposition="outside")
    
    # Save and show
    save_fig(fig, "intent_distribution")
    return fig


# -------------------------
# Main EDA Routine
# -------------------------

def main():
    print("Loading data from the database...")
    df = load_data()
    print("Data loaded. Number of records:", len(df))
    
    # Preview the DataFrame
    print(df.head())
    
    # Create directory for figures if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Create and save visualizations
    print("Plotting sales funnel classification...")
    fig_funnel = plot_sales_funnel_classification(df)
    fig_funnel.show()
    
    print("Plotting sentiment distribution by sales funnel stage...")
    fig1 = plot_sentiment_by_funnel(df)
    fig1.show()
    
    print("Plotting product mentions...")
    fig2 = plot_product_mentions(df)
    fig2.show()
    
    print("Plotting word count distribution...")
    fig3 = plot_word_count_distribution(df)
    fig3.show()
    
    print("Plotting business impact distribution...")
    fig5 = plot_business_impact_distribution(df)
    fig5.show()
    
    print("Plotting intent distribution...")
    fig6 = plot_intent_distribution(df)
    fig6.show()
    
    print("All visualizations complete. Check the 'figures' directory for saved outputs.")


if __name__ == "__main__":
    main()