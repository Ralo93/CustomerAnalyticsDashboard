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


# Configure logging with reduced duplication
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Remove existing loggers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                'logs/ml_insights_dashboard.log', 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Load environment variables
load_dotenv()

# Configuration
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", 60))  # Cache expiry in seconds

# Initialize Redis client with error handling
def create_redis_client():
    try:
        client = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            decode_responses=True,
            socket_timeout=2  # 2-second timeout
        )
        # Verify connection
        client.ping()
        logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return client
    except redis.ConnectionError:
        logger.error(f"Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return None
# Page config
st.set_page_config(
    page_title="ML Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

redis_client = create_redis_client()


# Function to fetch data from the database service with Redis caching
def fetch_sentences():
    # Cache key for sentences
    CACHE_KEY = 'ml_insights:sentences'
    
    try:
        # If Redis is available, try to use cache
        if redis_client:
            # Try to get cached data
            cached_data = redis_client.get(CACHE_KEY)
            if cached_data:
                logger.info("Retrieved sentences from Redis cache")
                return pd.DataFrame(json.loads(cached_data))
        
        # If no cache, fetch from database service
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{DB_SERVICE_URL}/sentences")
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch sentences: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return pd.DataFrame()
            
            sentences = response.json()
            df = pd.DataFrame(sentences)
            
            # Cache the data if Redis is available
            if redis_client:
                try:
                    redis_client.setex(
                        CACHE_KEY, 
                        CACHE_EXPIRY, 
                        json.dumps(df.to_dict(orient='records'))
                    )
                    logger.info(f"Cached {len(df)} sentences for {CACHE_EXPIRY} seconds")
                except Exception as cache_error:
                    logger.error(f"Failed to cache sentences: {cache_error}")
            
            logger.info(f"Successfully fetched {len(df)} sentences")
            return df
    
    except Exception as e:
        logger.error(f"Error fetching sentences: {str(e)}")
        st.error(f"Error fetching sentences: {str(e)}")
        return pd.DataFrame()

def verify_redis_caching():
    if not redis_client:
        st.warning("Redis is not connected. Caching is disabled.")
        return False
    
    try:
        # Test key
        test_key = 'ml_insights:test_cache'
        
        # Set a test value
        redis_client.setex(test_key, 10, json.dumps({'test': 'caching works'}))
        
        # Retrieve the value
        cached_test = redis_client.get(test_key)
        
        if cached_test:
            st.sidebar.success("✅ Redis Caching Verified")
        else:
            st.sidebar.warning("⚠️ Redis Caching Test Failed")
    
    except Exception as e:
        st.sidebar.error(f"❌ Redis Caching Error: {str(e)}")
        logger.error(f"Redis caching verification failed: {str(e)}")



def render_dashboard_content(sentences_df):
    # Check if we have data
    if not sentences_df.empty:
        # Log data summary
        logger.info("Processing sentence data")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_sentences = len(sentences_df)
            st.metric(
                label="Total Processed Sentences", 
                value=total_sentences
            )
            logger.info(f"Total processed sentences: {total_sentences}")
        
        with col2:
            high_priority_count = len(sentences_df[sentences_df['priority'] == 'high']) if 'priority' in sentences_df.columns else 0
            st.metric(
                label="High Priority Sentences", 
                value=high_priority_count
            )
            logger.info(f"High priority sentences: {high_priority_count}")
        
        with col3:
            normal_priority_count = len(sentences_df[sentences_df['priority'] == 'normal']) if 'priority' in sentences_df.columns else 0
            st.metric(
                label="Normal Priority Sentences", 
                value=normal_priority_count
            )
            logger.info(f"Normal priority sentences: {normal_priority_count}")
        
        # Create priority histogram
        if 'priority' in sentences_df.columns:
            st.subheader("Sentence Priority Distribution")
            
            # Count by priority
            priority_counts = sentences_df['priority'].value_counts().reset_index()
            priority_counts.columns = ['Priority', 'Count']
            
            # Log priority distribution
            logger.info("Sentence Priority Distribution:")
            for _, row in priority_counts.iterrows():
                logger.info(f"  {row['Priority']}: {row['Count']} sentences")
            
            # Create bar chart
            chart = alt.Chart(priority_counts).mark_bar().encode(
                x=alt.X('Priority:N', sort=None),
                y='Count:Q',
                color=alt.Color('Priority:N', scale=alt.Scale(
                    domain=['high', 'normal'],
                    range=['#ff4b4b', '#4b72ff']
                )),
                tooltip=['Priority', 'Count']
            ).properties(
                width=600,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        # Display sentences table with priority
        st.subheader("Processed Sentences")
        
        # Filter columns for display
        display_columns = ['id', 'text', 'priority', 'intent_type', 'sentiment']
        display_df = sentences_df[
            [col for col in display_columns if col in sentences_df.columns]
        ]
        
        # Add color coding to priority column
        if 'priority' in display_df.columns:
            def highlight_priority(val):
                color = 'background-color: #ffcccc' if val == 'high' else ''
                return color
            
            styled_df = display_df.style.applymap(
                highlight_priority, subset=['priority']
            )
            
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        # Show detailed view of high priority items
        if high_priority_count > 0 and 'priority' in sentences_df.columns:
            st.subheader("High Priority Sentences")
            high_priority_df = sentences_df[sentences_df['priority'] == 'high']
            
            logger.info("Detailed high priority sentences:")
            for _, row in high_priority_df.iterrows():
                with st.expander(f"ID: {row.get('id')} - {row.get('text', '')[:50]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Full Text:** {row.get('text', '')}")
                        st.markdown(f"**Intent Type:** {row.get('intent_type', 'Unknown')}")
                        st.markdown(f"**Sentiment:** {row.get('sentiment', 'Unknown')}")
                    
                    with col2:
                        st.markdown(f"**Priority Reasoning:** {row.get('priority_reasoning', '')}")
                    
                    # Log high priority sentence details
                    logger.info(f"  Sentence ID: {row.get('id')}")
                    logger.info(f"  Intent Type: {row.get('intent_type', 'Unknown')}")
                    logger.info(f"  Sentiment: {row.get('sentiment', 'Unknown')}")
    else:
        st.info("No sentences have been processed yet. Please check that your database service is running and sentences are being processed.")
        logger.warning("No sentences found in database")

  

def main():

  # Use Streamlit's native session state
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    # Verify Redis caching
    redis_verified = verify_redis_caching()

    # Sidebar controls
    st.sidebar.title("Dashboard Refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)", 
        min_value=3, 
        max_value=60, 
        value=10
    )


    st.title("ML Insights Dashboard")
    st.write("Real-time analytics of customer sentence processing")

    # Fetch data
    sentences_df = fetch_sentences()

    render_dashboard_content(sentences_df)

     # Manual refresh button
    if st.sidebar.button("Refresh Data Now"):
        st.session_state.refresh_count += 1
        logger.info(f"Manual refresh triggered. Refresh count: {st.session_state.refresh_count}")
        
        # Clear Redis cache if available
        if redis_client and redis_verified:
            try:
                redis_client.delete('ml_insights:sentences')
                logger.info("Cleared Redis cache for sentences")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")
        
        # Force a complete rerun
        st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        # Use Streamlit's built-in rerun mechanism
        st.empty()  # Add an empty placeholder to trigger refresh
        time.sleep(refresh_interval)
        st.session_state.refresh_count += 1
        logger.info(f"Auto-refresh triggered. Refresh count: {st.session_state.refresh_count}")
        
        # Clear Redis cache if available
        if redis_client and redis_verified:
            try:
                redis_client.delete('ml_insights:sentences')
                logger.info("Cleared Redis cache for sentences")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")
        
        # Force a complete rerun
        st.rerun()

    # Display refresh information
    st.sidebar.write(f"Total Refreshes: {st.session_state.refresh_count}")


# Run the main function
if __name__ == "__main__":
    main()