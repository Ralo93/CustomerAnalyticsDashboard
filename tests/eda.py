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
                "is_sales_funnel_relevant_confidence": label.is_sales_funnel_relevant_confidence if label else None,
                "sales_funnel_stage": label.sales_funnel_stage if label else None,
                "sales_funnel_confidence": label.sales_funnel_confidence if label else None,
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



def plot_product_mentions(df):
    """Simple bar chart: Count of product mentions for each product."""
    product_counts = {
        "MasterBlaster": df["mentions_masterblaster"].sum(),
        "FunPun": df["mentions_funpun"].sum(),
        "PowerPro": df["mentions_powerpro"].sum()
    }
    prod_df = pd.DataFrame(list(product_counts.items()), columns=["Product", "Mentions"])
    
    colors = {"PowerPro": "#2ca02c", "FunPun": "#1f77b4", "MasterBlaster": "#ff7f0e"}  # Green, Blue, Orange
    
    fig = px.bar(
        prod_df,
        x="Product",
        y="Mentions",
        title="Product Mentions",
        text="Mentions",
        color="Product",
        color_discrete_map=colors
    )
    
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
    
    fig.update_traces(textposition="outside")
    
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
    df_impact = df[df["business_impact"].notna()]
    impact_counts = df_impact["business_impact"].value_counts().reset_index()
    impact_counts.columns = ["Business Impact", "Count"]
    
    impact_counts["Business Impact"] = pd.Categorical(
        impact_counts["Business Impact"],
        categories=["Neutral", "Low", "Medium", "High", "Critical"],
        ordered=True
    )
    impact_counts = impact_counts.sort_values("Business Impact")
    
    fig = px.bar(
        impact_counts,
        x="Business Impact",
        y="Count",
        title="Business Impact Distribution",
        text="Count"
    )
    
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
    
    fig.update_traces(textposition="outside")
    
    save_fig(fig, "business_impact_distribution")
    return fig

import plotly.express as px

def plot_sales_funnel_classification(df):
    """Bar chart: Distribution of sales funnel stages."""
    df_funnel = df[df["sales_funnel_stage"].notna()]
    funnel_counts = df_funnel["sales_funnel_stage"].value_counts().reindex([
        "Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"
    ]).reset_index().fillna(0)
    funnel_counts.columns = ["Category", "Count"]

    fig = px.bar(
        funnel_counts,
        x="Category",
        y="Count",
        title="Sales Funnel Classification",
        text="Count",
        category_orders={"Category": ["Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"]}
    )

    fig.update_layout(
        title=dict(text="Sales Funnel Classification", font=dict(size=24)),
        xaxis=dict(
            title="Sales Funnel Stage",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            categoryorder="array",
            categoryarray=["Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"],
            showgrid=False
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=False
        ),
        height=600,  # Standardized height
        width=900   # Standardized width
    )

    fig.update_traces(textposition="outside")

    save_fig(fig, "sales_funnel_classification")
    return fig


def plot_sentiment_by_funnel(df):
    """Bar chart: Sentiment distribution grouped by sales funnel stage."""
    df_stage = df[df["sales_funnel_stage"].notna()]
    
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
        title="Sentiment by Sales Funnel Stage",
        labels={
            "sales_funnel_stage": "Sales Funnel Stage",
            "count": "Count",
            "sentiment": "Sentiment"
        },
        category_orders={"sales_funnel_stage": ["Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"]},
        color_discrete_map={
            "Positive": "green",
            "Neutral": "gray",
            "Negative": "red"
        }
    )

    fig.update_layout(
        title=dict(text="Sentiment by Sales Funnel Stage", font=dict(size=24)),
        xaxis=dict(
            title="Sales Funnel Stage",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            categoryorder="array",
            categoryarray=["Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"],
            showgrid=False
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=False
        ),
        legend=dict(
            title="Sentiment",
            font=dict(size=16)
        ),
        height=600,  # Standardized height
        width=900    # Standardized width
    )

    save_fig(fig, "sentiment_by_funnel")
    return fig



def plot_intent_distribution(df):
    """Bar chart: Distribution of communication intents."""
    df_intent = df[df["intent"].notna()]
    intent_counts = df_intent["intent"].value_counts().reindex([
        "Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"
    ]).reset_index().fillna(0)
    intent_counts.columns = ["Category", "Count"]

    fig = px.bar(
        intent_counts,
        x="Category",
        y="Count",
        title="Intent Distribution",
        text="Count",
        category_orders={"Category": ["Awareness", "Interest", "Consideration", "Evaluation", "Intent", "Purchase"]}
    )

    fig.update_layout(
        title=dict(text="Intent Distribution", font=dict(size=24)),
        xaxis=dict(title="Category", title_font=dict(size=18), tickfont=dict(size=16), showgrid=False),
        yaxis=dict(title="Count", title_font=dict(size=18), tickfont=dict(size=16), showgrid=False),
        height=500,
        width=800
    )

    fig.update_traces(textposition="outside")

    save_fig(fig, "intent_distribution")
    return fig


def plot_sales_funnel_confidence_scores(df):
    """Box plot: Distribution of sales funnel related confidence scores."""
    # Define sales funnel confidence columns
    funnel_confidence_cols = [
        'sales_funnel_confidence',
        'is_sales_funnel_relevant_confidence'
    ]
    
    # Only use columns that exist in the dataframe
    confidence_cols = [col for col in funnel_confidence_cols if col in df.columns]
    
    if not confidence_cols:
        print("No sales funnel confidence score columns found in the dataframe.")
        return None
    
    # Create a copy of the dataframe with only the available confidence columns
    df_conf = df[confidence_cols].copy()
    
    # Handle the special case for sales_funnel_confidence
    # Only use sales_funnel_confidence where is_sales_funnel_relevant is True
    if 'sales_funnel_confidence' in df_conf.columns and 'is_sales_funnel_relevant' in df.columns:
        mask = df['is_sales_funnel_relevant'] != True
        df_conf.loc[mask, 'sales_funnel_confidence'] = None
    
    # Melt the dataframe to get confidence scores in long format
    melted_df = pd.melt(
        df_conf, 
        value_vars=confidence_cols,
        var_name='Confidence Type', 
        value_name='Confidence Score'
    )
    
    # Drop rows with null confidence scores
    melted_df = melted_df.dropna(subset=['Confidence Score'])
    
    if len(melted_df) == 0:
        print("No non-null sales funnel confidence scores found in the dataframe.")
        return None
    
    # Clean up the labels for display
    melted_df['Confidence Type'] = melted_df['Confidence Type'].apply(
        lambda x: 'Funnel Stage' if x == 'sales_funnel_confidence' else 'Funnel Relevance'
    )
    
    # Create the box plot with improved visual style
    fig = px.box(
        melted_df,
        x='Confidence Type',
        y='Confidence Score',
        title='Sales Funnel Confidence Scores',
        points='all',  # Show all points
        color='Confidence Type',
        notched=True,  # Add notches to better show confidence interval of median
        boxmode='overlay'  # Overlay boxes for better comparison
    )
    
    fig.update_layout(
        title=dict(
            text='Sales Funnel Confidence Scores',
            font=dict(size=24)
        ),
        xaxis=dict(
            title='Prediction Type',
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=False
        ),
        yaxis=dict(
            title='Confidence Score',
            title_font=dict(size=18),
            tickfont=dict(size=16),
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(220,220,220,0.5)'  # Lighter grid for better readability
        ),
        height=600,
        width=900,
        boxgap=0.3,  # More space between boxes
        plot_bgcolor='white'  # White background for better contrast
    )
    
    # Add horizontal reference lines at key confidence thresholds
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="rgba(0,0,0,0.3)"),
        y0=0.7, y1=0.7, x0=-0.5, x1=len(confidence_cols)-0.5,
        layer="below"
    )
    
    fig.add_annotation(
        x=0, y=0.73, 
        text="High confidence (>0.7)",
        showarrow=False,
        font=dict(size=12)
    )
    
    # Add a note about data completeness
    for i, col in enumerate(confidence_cols):
        display_name = 'Funnel Stage' if col == 'sales_funnel_confidence' else 'Funnel Relevance'
        non_null = df_conf[col].count()
        total = len(df_conf)
        pct = (non_null / total) * 100
        
        fig.add_annotation(
            x=display_name,
            y=0.05,
            text=f"Data available: {non_null}/{total} ({pct:.1f}%)",
            showarrow=False,
            font=dict(size=12)
        )
    
    save_fig(fig, "sales_funnel_confidence_scores")
    return fig

def plot_other_confidence_scores(df):
    """Box plot: Distribution of sentiment, intent, and business impact confidence scores."""
    # Define non-sales funnel confidence columns
    other_confidence_cols = [
        'sentiment_confidence', 
        'business_impact_confidence', 
        'intent_confidence'
    ]
    
    # Only use columns that exist in the dataframe
    confidence_cols = [col for col in other_confidence_cols if col in df.columns]
    
    if not confidence_cols:
        print("No sentiment, intent, or business impact confidence score columns found in the dataframe.")
        return None
    
    # Create a copy of the dataframe with only the available confidence columns
    df_conf = df[confidence_cols].copy()
    
    # Melt the dataframe to get confidence scores in long format
    melted_df = pd.melt(
        df_conf, 
        value_vars=confidence_cols,
        var_name='Confidence Type', 
        value_name='Confidence Score'
    )
    
    # Drop rows with null confidence scores
    melted_df = melted_df.dropna(subset=['Confidence Score'])
    
    if len(melted_df) == 0:
        print("No non-null sentiment, intent, or business impact confidence scores found in the dataframe.")
        return None
    
    # Clean up the labels for display
    melted_df['Confidence Type'] = melted_df['Confidence Type'].apply(
        lambda x: x.replace('_confidence', '').replace('_', ' ').title()
    )
    
    # Create the box plot with improved visual style
    fig = px.box(
        melted_df,
        x='Confidence Type',
        y='Confidence Score',
        title='Sentiment, Intent, and Business Impact Confidence Scores',
        points='all',  # Show all points
        color='Confidence Type',
        notched=True,  # Add notches to better show confidence interval of median
        boxmode='overlay'  # Overlay boxes for better comparison
    )
    
    fig.update_layout(
        title=dict(
            text='Sentiment, Intent, and Business Impact Confidence Scores',
            font=dict(size=24)
        ),
        xaxis=dict(
            title='Prediction Type',
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=False
        ),
        yaxis=dict(
            title='Confidence Score',
            title_font=dict(size=18),
            tickfont=dict(size=16),
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(220,220,220,0.5)'  # Lighter grid for better readability
        ),
        height=600,
        width=900,
        boxgap=0.3,  # More space between boxes
        plot_bgcolor='white'  # White background for better contrast
    )
    
    # Add horizontal reference lines at key confidence thresholds
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="rgba(0,0,0,0.3)"),
        y0=0.7, y1=0.7, x0=-0.5, x1=len(confidence_cols)-0.5,
        layer="below"
    )
    
    fig.add_annotation(
        x=1, y=0.73, 
        text="High confidence (>0.7)",
        showarrow=False,
        font=dict(size=12)
    )
    
    # Add a note about data completeness
    for i, col in enumerate(confidence_cols):
        display_name = col.replace('_confidence', '').replace('_', ' ').title()
        non_null = df_conf[col].count()
        total = len(df_conf)
        pct = (non_null / total) * 100
        
        fig.add_annotation(
            x=display_name,
            y=0.05,
            text=f"Data available: {non_null}/{total} ({pct:.1f}%)",
            showarrow=False,
            font=dict(size=12)
        )
    
    save_fig(fig, "other_confidence_scores")
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
    print("Plotting confidence score distribution...")
    #fig_funnel = plot_other_confidence_scores(df)
    #fig_funnel.show()

    fig_funnel = plot_sales_funnel_confidence_scores(df)
    fig_funnel.show()
    
    # Create and save visualizations
    #print("Plotting sales funnel classification...")
    #fig_funnel = plot_sales_funnel_classification(df)
    #fig_funnel.show()
    
    #print("Plotting sentiment distribution by sales funnel stage...")
    #fig1 = plot_sentiment_by_funnel(df)
    #fig1.show()
    
    print("Plotting product mentions...")
    #fig2 = plot_product_mentions(df)
    #fig2.show()
    
    print("Plotting word count distribution...")
    #fig3 = plot_word_count_distribution(df)
    #fig3.show()
    
    print("Plotting business impact distribution...")
    #fig5 = plot_business_impact_distribution(df)
    #fig5.show()
    
    print("Plotting intent distribution...")
    #fig6 = plot_intent_distribution(df)
    #fig6.show()
    
    print("All visualizations complete. Check the 'figures' directory for saved outputs.")


if __name__ == "__main__":
    main()