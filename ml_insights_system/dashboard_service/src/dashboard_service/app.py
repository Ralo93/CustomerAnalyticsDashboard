import threading
import time
import streamlit as st
import pandas as pd
import altair as alt
import httpx
import os
import asyncio
import aio_pika
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://localhost:8001")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed")

# Create a session state to track updates
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = time.time()
    st.session_state.update_available = False
    st.session_state.refresh_count = 0

# Page config
st.set_page_config(
    page_title="ML Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("ML Insights Dashboard")
st.write("Real-time analytics of customer sentence processing")

last_updated_time = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_updated))
st.write(f"Last updated: {last_updated_time}")

# RabbitMQ listener for aio_pika
async def async_rabbitmq_listener():
    """Async listener for RabbitMQ using aio_pika"""
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        
        # Create channel
        channel = await connection.channel()
        
        # Declare the queue
        queue = await channel.declare_queue(
            PROCESSED_QUEUE_NAME,
            durable=True
        )
        
        print(f"Connected to RabbitMQ, listening on queue: {PROCESSED_QUEUE_NAME}")
        
        # Define callback for processing messages
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    body = message.body.decode()
                    print(f"Received message: {body}")
                    
                    # Set flag for dashboard to refresh
                    st.session_state.update_available = True
                    st.session_state.refresh_count += 1
                    print(f"Set update_available to True, refresh count: {st.session_state.refresh_count}")
                except Exception as e:
                    print(f"Error processing message: {str(e)}")
        
        # Start consuming
        await queue.consume(process_message)
        
        # Keep the connection alive
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error in async RabbitMQ listener: {str(e)}")
        # Try to reconnect after a delay
        await asyncio.sleep(5)
        asyncio.create_task(async_rabbitmq_listener())

# Thread wrapper for the async listener
def rabbitmq_listener_thread():
    """Thread wrapper for the async RabbitMQ listener"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_rabbitmq_listener())
    except Exception as e:
        print(f"Error in RabbitMQ listener thread: {str(e)}")
    finally:
        loop.close()

# Start the RabbitMQ listener thread when the app starts
if 'listener_thread' not in st.session_state:
    st.session_state.listener_thread = threading.Thread(target=rabbitmq_listener_thread, daemon=True)
    st.session_state.listener_thread.start()
    print("Started RabbitMQ listener thread")
    
# Function to fetch data from the database service
@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_sentences():
    try:
        # Since Streamlit runs synchronously, we need to use the synchronous version of httpx
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{DB_SERVICE_URL}/sentences")
            
            if response.status_code != 200:
                st.error(f"Failed to fetch sentences: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            sentences = response.json()
            # Convert to pandas DataFrame for easier manipulation
            df = pd.DataFrame(sentences)
            return df
    except Exception as e:
        st.error(f"Error fetching sentences: {str(e)}")
        return pd.DataFrame()

# Create status indicators in sidebar
st.sidebar.title("Dashboard Status")
st.sidebar.write(f"Last Updated: {last_updated_time}")
st.sidebar.write(f"Refresh Count: {st.session_state.refresh_count}")
rabbitmq_status = st.sidebar.empty()
rabbitmq_status.info(f"Connecting to RabbitMQ queue: {PROCESSED_QUEUE_NAME}")

# Configure auto-refresh settings
auto_refresh = st.sidebar.checkbox("Enable Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)

# Check if we need to update (either timer or RabbitMQ notification)
current_time = time.time()
update_needed = False

# Check for RabbitMQ update notification
if st.session_state.update_available:
    st.sidebar.success("Received update notification!")
    st.session_state.update_available = False
    update_needed = True

# Check for time-based refresh
time_elapsed = current_time - st.session_state.last_updated
if auto_refresh and time_elapsed >= refresh_interval:
    update_needed = True
    
# Apply the update if needed
if update_needed:
    st.session_state.last_updated = current_time
    fetch_sentences.clear()
    
# Fetch data
sentences_df = fetch_sentences()

# Check if we have data
if not sentences_df.empty:
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Processed Sentences", 
            value=len(sentences_df)
        )
    
    with col2:
        high_priority_count = len(sentences_df[sentences_df['priority'] == 'high']) if 'priority' in sentences_df.columns else 0
        st.metric(
            label="High Priority Sentences", 
            value=high_priority_count
        )
    
    with col3:
        normal_priority_count = len(sentences_df[sentences_df['priority'] == 'normal']) if 'priority' in sentences_df.columns else 0
        st.metric(
            label="Normal Priority Sentences", 
            value=normal_priority_count
        )
    
    # Create priority histogram
    if 'priority' in sentences_df.columns:
        st.subheader("Sentence Priority Distribution")
        
        # Count by priority
        priority_counts = sentences_df['priority'].value_counts().reset_index()
        priority_counts.columns = ['Priority', 'Count']
        
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
        
        for _, row in high_priority_df.iterrows():
            with st.expander(f"ID: {row.get('id')} - {row.get('text', '')[:50]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Full Text:** {row.get('text', '')}")
                    st.markdown(f"**Intent Type:** {row.get('intent_type', 'Unknown')}")
                    st.markdown(f"**Sentiment:** {row.get('sentiment', 'Unknown')}")
                
                with col2:
                    st.markdown(f"**Priority Reasoning:** {row.get('priority_reasoning', '')}")
else:
    st.info("No sentences have been processed yet. Please check that your database service is running and sentences are being processed.")

# Add manual refresh button
if st.button("Refresh Data Now"):
    st.session_state.refresh_count += 1
    st.rerun()

# Simple auto-refresh as a fallback
if auto_refresh:
    time_remaining = refresh_interval - int(time_elapsed)
    st.sidebar.write(f"Next auto-refresh in {time_remaining} seconds")
    
    # Force refresh when timer expires
    if time_elapsed >= refresh_interval:
        st.rerun()