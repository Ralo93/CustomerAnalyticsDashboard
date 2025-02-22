import time
import streamlit as st
import pandas as pd
import altair as alt
import httpx
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('ml_insights_dashboard')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/ml_insights_dashboard.log', 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Load environment variables
load_dotenv()

# Configuration
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
BRIDGE_API_URL = os.getenv("BRIDGE_API_URL", "http://localhost:8005")

# Page config
st.set_page_config(
    page_title="ML Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

# Setup session state
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = time.time()
    st.session_state.last_message_count = 0
    st.session_state.bridge_status = "Unknown"

# Header
st.title("ML Insights Dashboard")
st.write("Real-time analytics of customer sentence processing")

# Last updated display
last_updated_time = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_updated))
st.write(f"Last updated: {last_updated_time}")

# Function to check bridge service for new messages
@st.cache_data(ttl=3)  # Cache for 3 seconds
def check_bridge_service():
    try:
        logger.info("Checking bridge service status")
        with httpx.Client(timeout=5.0) as client:
            # Check health
            health_response = client.get(f"{BRIDGE_API_URL}/health")
            
            if health_response.status_code != 200:
                logger.warning(f"Bridge service health check failed. Status code: {health_response.status_code}")
                return "Unavailable", 0, None
            
            health_data = health_response.json()
            logger.debug(f"Bridge service health data: {health_data}")
            
            # Get messages
            messages_response = client.get(f"{BRIDGE_API_URL}/messages")
            if messages_response.status_code != 200:
                logger.warning(f"Failed to fetch messages. Status code: {messages_response.status_code}")
                return health_data.get("status", "Degraded"), 0, None
            
            messages = messages_response.json()
            logger.info(f"Retrieved {len(messages)} messages from bridge service")
            
            return health_data.get("status", "Unknown"), len(messages), messages
    except Exception as e:
        logger.error(f"Error checking bridge service: {str(e)}", exc_info=True)
        return "Error", 0, None

# Function to fetch data from the database service
@st.cache_data(ttl=5)  # Cache for 5 seconds
def fetch_sentences():
    try:
        logger.info("Fetching sentences from database service")
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{DB_SERVICE_URL}/sentences")
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch sentences: {response.status_code} - {response.text}"
                logger.error(error_msg)
                st.error(error_msg)
                return pd.DataFrame()
            
            sentences = response.json()
            df = pd.DataFrame(sentences)
            
            logger.info(f"Successfully fetched {len(df)} sentences")
            logger.debug(f"Sentence data columns: {df.columns.tolist()}")
            
            return df
    except Exception as e:
        error_msg = f"Error fetching sentences: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return pd.DataFrame()

# Logging setup and initial messages
logger.info("ML Insights Dashboard Starting")
logger.info(f"Database Service URL: {DB_SERVICE_URL}")
logger.info(f"Bridge API URL: {BRIDGE_API_URL}")

# Check for new messages from bridge service
bridge_status, message_count, messages = check_bridge_service()

# Status display in sidebar
st.sidebar.title("Dashboard Status")
if bridge_status == "healthy":
    st.sidebar.success(f"✅ Bridge connected to RabbitMQ")
    logger.info("Bridge service status: Healthy")
elif bridge_status == "degraded":
    st.sidebar.warning(f"⚠️ Bridge service degraded")
    logger.warning("Bridge service status: Degraded")
else:
    st.sidebar.error(f"❌ Bridge service unavailable")
    logger.error("Bridge service status: Unavailable")

# Also show auto-refresh controls in sidebar
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 
                                   min_value=3, max_value=60, value=10)

# Check if we need to update
update_needed = False
current_time = time.time()

# Update if new messages are available
if message_count > st.session_state.last_message_count:
    update_needed = True
    st.session_state.last_message_count = message_count
    
    # Show notification about new messages
    if messages and len(messages) > 0:
        latest = messages[-1]  # Most recent message
        sentence_id = latest.get('sentence_id', 'unknown')
        priority = latest.get('priority', 'unknown')
        st.sidebar.info(f"New sentence processed: {sentence_id[:8]}... with {priority} priority")
        
        # Log new message details
        logger.info(f"New message received - Sentence ID: {sentence_id}, Priority: {priority}")

# Or update based on timer
time_elapsed = current_time - st.session_state.last_updated
if auto_refresh and time_elapsed >= refresh_interval:
    update_needed = True
    logger.debug(f"Auto-refresh triggered. Time elapsed: {time_elapsed}, Interval: {refresh_interval}")

# Apply the update if needed
if update_needed:
    st.session_state.last_updated = current_time
    fetch_sentences.clear()
    logger.info("Dashboard data update triggered")
    
# Fetch data
sentences_df = fetch_sentences()

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

# Add manual refresh button
if st.button("Refresh Data Now"):
    logger.info("Manual refresh triggered")
    st.session_state.last_updated = time.time()
    fetch_sentences.clear()
    check_bridge_service.clear()
    st.rerun()

# Display countdown for next refresh
if auto_refresh:
    time_remaining = max(0, refresh_interval - int(time_elapsed))
    st.sidebar.write(f"Next refresh in {time_remaining} seconds")
    
    # Force refresh when timer expires
    if time_elapsed >= refresh_interval:
        logger.debug("Auto-refresh timer expired. Rerunning.")
        st.rerun()

# Log dashboard session completion
logger.info("ML Insights Dashboard session completed")