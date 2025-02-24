import time
import streamlit as st
import pandas as pd
import altair as alt
import httpx
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import redis
import json
import numpy as np
from typing import Dict, Optional

# Keep existing logging setup
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                'logs/ml_insights_dashboard.log', 
                maxBytes=10*1024*1024,
                backupCount=5
            )
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
load_dotenv()

# Configuration
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", 60))

# Set up page configuration
st.set_page_config(page_title="ML Insights Dashboard", page_icon="📊", layout="wide")

# Redis client initialization
def create_redis_client():
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=2
        )
        client.ping()
        logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return client
    except redis.ConnectionError:
        logger.error(f"Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return None

redis_client = create_redis_client()

def verify_redis_caching():
    """Verify Redis connection and caching"""
    if not redis_client:
        st.warning("Redis is not connected. Caching is disabled.")
        return False
    
    try:
        test_key = 'ml_insights:test_cache'
        redis_client.setex(test_key, 10, json.dumps({'test': 'caching works'}))
        cached_test = redis_client.get(test_key)
        
        if cached_test:
            st.sidebar.success("✅ Redis Caching Verified")
            return True
        else:
            st.sidebar.warning("⚠️ Redis Caching Test Failed")
            return False
    
    except Exception as e:
        st.sidebar.error(f"❌ Redis Caching Error: {str(e)}")
        logger.error(f"Redis caching verification failed: {str(e)}")
        return False

# Generic data fetching function with Redis caching
def fetch_data(endpoint: str, cache_key: str):
    """Fetch data from API with Redis caching"""
    try:
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved {endpoint} data from Redis cache")
                return pd.DataFrame(json.loads(cached_data))
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{DB_SERVICE_URL}/analytics/{endpoint}")
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch {endpoint} data: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return pd.DataFrame()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            if redis_client:
                try:
                    redis_client.setex(
                        cache_key,
                        CACHE_EXPIRY,
                        json.dumps(df.to_dict(orient='records'))
                    )
                    logger.info(f"Cached {endpoint} data for {CACHE_EXPIRY} seconds")
                except Exception as cache_error:
                    logger.error(f"Failed to cache {endpoint} data: {cache_error}")
            
            return df
    except Exception as e:
        logger.error(f"Error fetching {endpoint} data: {str(e)}")
        st.error(f"Error fetching {endpoint} data: {str(e)}")
        return pd.DataFrame()

def fetch_sentences_by_filter(filters: Dict[str, str], limit: int = 10):
    """Fetch sentences matching specific filter criteria"""
    try:
        # Remove None values from filters
        filters = {k: v for k, v in filters.items() if v is not None}
        
        # Create cache key based on filters
        cache_key = f"ml_insights:sentences:{hash(frozenset(filters.items()))}"
        
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved filtered sentences from Redis cache")
                return json.loads(cached_data)
        
        # Build query parameters
        params = {**filters, 'limit': limit}
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{DB_SERVICE_URL}/analytics/sentences-by-filter", params=params)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch sentences: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return []
            
            data = response.json()
            
            if redis_client:
                try:
                    redis_client.setex(
                        cache_key,
                        CACHE_EXPIRY,
                        json.dumps(data)
                    )
                    logger.info(f"Cached {len(data)} filtered sentences for {CACHE_EXPIRY} seconds")
                except Exception as cache_error:
                    logger.error(f"Failed to cache filtered sentences: {cache_error}")
            
            return data
    except Exception as e:
        logger.error(f"Error fetching sentences: {str(e)}")
        st.error(f"Error fetching sentences: {str(e)}")
        return []

def display_sentence_details(sentences_data):
    """Display sentence details in an expandable format"""
    if not sentences_data:
        st.info("No sentences match the selected criteria.")
        return
    
    st.write(f"### Related Sentences ({len(sentences_data)})")
    
    for sentence in sentences_data:
        with st.expander(f"{sentence['text'][:80]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Full Text:**")
                st.write(sentence['text'])
                st.write(f"**Created:** {sentence['created_at']}")
            
            with col2:
                st.write("**Attributes:**")
                st.write(f"- **Sales Funnel Stage:** {sentence['sales_funnel_stage']}")
                st.write(f"- **Sentiment:** {sentence['sentiment']}")
                st.write(f"- **Business Impact:** {sentence['business_impact']}")
                st.write(f"- **Intent:** {sentence['intent']}")

# Visualization functions with click interaction
def render_sentiment_distribution(df):
    """Visualize sentiment distribution per sales funnel stage with interactive elements"""
    if df.empty:
        st.warning("No sentiment distribution data available")
        return

    st.subheader("Sentiment Distribution by Sales Funnel Stage")
    
    # Create normalized stacked bar chart
    df_melted = pd.melt(
        df,
        id_vars=['sales_funnel_stage'],
        value_vars=['positive_count', 'negative_count', 'neutral_count'],
        var_name='sentiment_type',
        value_name='count'
    )
    
    # Map sentiment types to display values
    df_melted['sentiment_display'] = df_melted['sentiment_type'].map({
        'positive_count': 'Positive',
        'negative_count': 'Negative',
        'neutral_count': 'Neutral'
    })

    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('sales_funnel_stage:N', title='Sales Funnel Stage'),
        y=alt.Y('count:Q', stack=True, title='Count'),
        color=alt.Color('sentiment_type:N', scale=alt.Scale(
            domain=['positive_count', 'negative_count', 'neutral_count'],
            range=['#2ecc71', '#e74c3c', '#95a5a6']
        )),
        tooltip=[
            alt.Tooltip('sales_funnel_stage:N', title='Stage'),
            alt.Tooltip('sentiment_display:N', title='Sentiment'),
            alt.Tooltip('count:Q', format=',', title='Count')
        ]
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Display the chart
    selected_point = st.altair_chart(chart, use_container_width=True)
    
    # Filter selection controls
    col1, col2 = st.columns(2)
    with col1:
        selected_stage = st.selectbox(
            "Select Sales Funnel Stage", 
            options=["All"] + sorted(df['sales_funnel_stage'].unique().tolist())
        )
    
    with col2:
        selected_sentiment = st.selectbox(
            "Select Sentiment",
            options=["All", "Positive", "Negative", "Neutral"]
        )
    
    # Fetch sentences based on selections
    filters = {}
    if selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    if selected_sentiment != "All":
        filters['sentiment'] = selected_sentiment
    
    if filters:
        sentences = fetch_sentences_by_filter(filters)
        display_sentence_details(sentences)

def render_impact_scores(df):
    """Visualize average impact scores per stage"""
    if df.empty:
        st.warning("No impact score data available")
        return

    st.subheader("Average Impact Scores by Sales Funnel Stage")
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('sales_funnel_stage:N', title='Sales Funnel Stage'),
        y=alt.Y('average_impact:Q', title='Average Impact Score'),
        color=alt.Color('average_impact:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=[
            alt.Tooltip('sales_funnel_stage:N', title='Stage'),
            alt.Tooltip('average_impact:Q', format='.2f', title='Avg Impact'),
            alt.Tooltip('count:Q', format=',', title='Count')
        ]
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection
    selected_stage = st.selectbox(
        "Select Stage to View Sentences",
        options=["All"] + sorted(df['sales_funnel_stage'].unique().tolist()),
        key="impact_stage_select"
    )
    
    if selected_stage and selected_stage != "All":
        sentences = fetch_sentences_by_filter({'sales_funnel_stage': selected_stage})
        display_sentence_details(sentences)

def render_sentiment_impact(df):
    """Visualize sentiment and business impact correlation"""
    if df.empty:
        st.warning("No sentiment impact data available")
        return

    st.subheader("Sentiment and Business Impact Correlation")
    
    # Create heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('sentiment:N', title='Sentiment'),
        y=alt.Y('business_impact:N', title='Business Impact'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['sentiment', 'business_impact', 'count']
    ).properties(
        width=500,
        height=300
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection controls
    col1, col2 = st.columns(2)
    with col1:
        selected_sentiment = st.selectbox(
            "Select Sentiment", 
            options=["All"] + sorted(df['sentiment'].unique().tolist()),
            key="sentiment_impact_select"
        )
    
    with col2:
        selected_impact = st.selectbox(
            "Select Business Impact",
            options=["All"] + sorted(df['business_impact'].unique().tolist()),
            key="business_impact_select"
        )
    
    # Fetch sentences based on selections
    filters = {}
    if selected_sentiment != "All":
        filters['sentiment'] = selected_sentiment
    
    if selected_impact != "All":
        filters['business_impact'] = selected_impact
    
    if filters:
        sentences = fetch_sentences_by_filter(filters)
        display_sentence_details(sentences)

def render_funnel_metrics(df):
    """Visualize funnel metrics"""
    if df.empty:
        st.warning("No funnel metrics data available")
        return

    st.subheader("Sales Funnel Metrics")
    
    # Create funnel chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('count:Q', title='Count'),
        y=alt.Y('stage:N', 
                title='Funnel Stage',
                sort=alt.EncodingSortField(field='count', order='descending')),
        color=alt.Color('percentage:Q', scale=alt.Scale(scheme='blues')),
        tooltip=[
            'stage',
            alt.Tooltip('count:Q', format=','),
            alt.Tooltip('percentage:Q', format='.1f')
        ]
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection
    selected_stage = st.selectbox(
        "Select Stage to View Sentences",
        options=["All"] + sorted(df['stage'].unique().tolist()),
        key="funnel_stage_select"
    )
    
    if selected_stage and selected_stage != "All":
        sentences = fetch_sentences_by_filter({'sales_funnel_stage': selected_stage})
        display_sentence_details(sentences)

def render_intent_distribution(df):
    """Visualize communication intent distribution"""
    if df.empty:
        st.warning("No intent distribution data available")
        return

    st.subheader("Communication Intent Distribution")
    
    # Create donut chart
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta('count:Q'),
        color=alt.Color('intent:N', scale=alt.Scale(scheme='category20')),
        tooltip=[
            'intent',
            alt.Tooltip('count:Q', format=','),
            alt.Tooltip('percentage:Q', format='.1f')
        ]
    ).properties(
        width=500,
        height=500
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection
    selected_intent = st.selectbox(
        "Select Intent to View Sentences",
        options=["All"] + sorted(df['intent'].unique().tolist()),
        key="intent_select"
    )
    
    if selected_intent and selected_intent != "All":
        sentences = fetch_sentences_by_filter({'intent': selected_intent})
        display_sentence_details(sentences)

def render_high_impact_sentiment(df):
    """Visualize high impact and positive sentiment per stage"""
    if df.empty:
        st.warning("No high impact sentiment data available")
        return

    st.subheader("High Impact Positive Sentiment by Stage")
    
    # Create chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('sales_funnel_stage:N', title='Sales Funnel Stage'),
        y=alt.Y('positive_high_impact_count:Q', title='High Impact Positive Count'),
        color=alt.Color('percentage:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=[
            'sales_funnel_stage', 
            alt.Tooltip('positive_high_impact_count:Q', title='High Impact Positive Count'),
            alt.Tooltip('total_count:Q', title='Total Count'),
            alt.Tooltip('percentage:Q', format='.1f', title='Percentage')
        ]
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection
    selected_stage = st.selectbox(
        "Select Stage to View High Impact Positive Sentences",
        options=["All"] + sorted(df['sales_funnel_stage'].unique().tolist()),
        key="high_impact_stage_select"
    )
    
    if selected_stage and selected_stage != "All":
        sentences = fetch_sentences_by_filter({
            'sales_funnel_stage': selected_stage,
            'sentiment': 'Positive',
            'business_impact': 'High'
        })
        display_sentence_details(sentences)

def render_stage_intent_alignment(df):
    """Visualize alignment between sales funnel stages and intents"""
    if df.empty:
        st.warning("No stage-intent alignment data available")
        return

    st.subheader("Sales Funnel Stage and Intent Alignment")
    
    # Create heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('sales_funnel_stage:N', title='Sales Funnel Stage'),
        y=alt.Y('intent:N', title='Intent'),
        color=alt.Color('alignment_score:Q', scale=alt.Scale(scheme='redblue')),
        tooltip=[
            'sales_funnel_stage',
            'intent',
            alt.Tooltip('count:Q', format=','),
            alt.Tooltip('alignment_score:Q', format='.2f')
        ]
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection controls
    col1, col2 = st.columns(2)
    with col1:
        selected_stage = st.selectbox(
            "Select Sales Funnel Stage", 
            options=["All"] + sorted(df['sales_funnel_stage'].unique().tolist()),
            key="alignment_stage_select"
        )
    
    with col2:
        selected_intent = st.selectbox(
            "Select Intent",
            options=["All"] + sorted(df['intent'].unique().tolist()),
            key="alignment_intent_select"
        )
    
    # Fetch sentences based on selections
    filters = {}
    if selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    if selected_intent != "All":
        filters['intent'] = selected_intent
    
    if filters:
        sentences = fetch_sentences_by_filter(filters)
        display_sentence_details(sentences)

def render_impact_information(df):
    """Visualize relationship between information type and business impact"""
    if df.empty:
        st.warning("No impact-information data available")
        return

    st.subheader("Information Type and Business Impact Relationship")
    
    # Create grouped bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('information_type:N', title='Information Type'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('business_impact:N', scale=alt.Scale(scheme='tableau10')),
        tooltip=['information_type', 'business_impact', 'count']
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection controls
    col1, col2 = st.columns(2)
    with col1:
        selected_info_type = st.selectbox(
            "Select Information Type", 
            options=["All"] + sorted(df['information_type'].unique().tolist()),
            key="info_type_select"
        )
    
    with col2:
        selected_impact = st.selectbox(
            "Select Business Impact",
            options=["All"] + sorted(df['business_impact'].unique().tolist()),
            key="info_impact_select"
        )
    
    # Fetch sentences based on selections
    filters = {}
    if selected_info_type != "All":
        filters['intent'] = selected_info_type
    
    if selected_impact != "All":
        filters['business_impact'] = selected_impact
    
    if filters:
        sentences = fetch_sentences_by_filter(filters)
        display_sentence_details(sentences)

def main():
    # Initialize session state
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0

    # Verify Redis caching
    redis_verified = verify_redis_caching()

    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=15
    )

    # Main content
    st.title("Customer Interaction Analytics Dashboard")
    st.write("Real-time analytics of customer interactions with interactive data exploration")

    # Create tabs for different visualizations
    tabs = st.tabs([
        "Sentiment Distribution",
        "Impact Scores",
        "Sentiment Impact",
        "Funnel Metrics",
        "Intent Distribution",
        "High Impact Sentiment",
        "Stage Intent Alignment",
        "Impact Information"
    ])

    # Fetch and render data for each tab
    with tabs[0]:
        df_sentiment = fetch_data("sentiment-distribution", "ml_insights:sentiment_dist")
        render_sentiment_distribution(df_sentiment)

    with tabs[1]:
        df_impact = fetch_data("impact-scores", "ml_insights:impact_scores")
        render_impact_scores(df_impact)

    with tabs[2]:
        df_sent_impact = fetch_data("sentiment-impact", "ml_insights:sentiment_impact")
        render_sentiment_impact(df_sent_impact)

    with tabs[3]:
        df_funnel = fetch_data("funnel-metrics", "ml_insights:funnel_metrics")
        render_funnel_metrics(df_funnel)

    with tabs[4]:
        df_intent = fetch_data("intent-distribution", "ml_insights:intent_dist")
        render_intent_distribution(df_intent)

    with tabs[5]:
        df_high_impact = fetch_data("high-impact-sentiment", "ml_insights:high_impact")
        render_high_impact_sentiment(df_high_impact)

    with tabs[6]:
        df_alignment = fetch_data("stage-intent-alignment", "ml_insights:alignment")
        render_stage_intent_alignment(df_alignment)

    with tabs[7]:
        df_info_impact = fetch_data("impact-information", "ml_insights:info_impact")
        render_impact_information(df_info_impact)

    # Manual refresh button
    if st.sidebar.button("Refresh Data Now"):
        st.session_state.refresh_count += 1
        logger.info(f"Manual refresh triggered. Refresh count: {st.session_state.refresh_count}")
        
        # Clear all Redis caches
        if redis_client and redis_verified:
            try:
                redis_client.delete(
                    "ml_insights:sentiment_dist",
                    "ml_insights:impact_scores",
                    "ml_insights:sentiment_impact",
                    "ml_insights:funnel_metrics",
                    "ml_insights:intent_dist",
                    "ml_insights:high_impact",
                    "ml_insights:alignment",
                    "ml_insights:info_impact"
                )
                logger.info("Cleared all Redis caches")
            except Exception as e:
                logger.error(f"Failed to clear Redis caches: {e}")
        
        st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        st.empty()
        time.sleep(refresh_interval)
        st.session_state.refresh_count += 1
        logger.info(f"Auto-refresh triggered. Refresh count: {st.session_state.refresh_count}")
        st.rerun()

    # Display refresh information
    st.sidebar.write(f"Total Refreshes: {st.session_state.refresh_count}")

if __name__ == "__main__":
    main()