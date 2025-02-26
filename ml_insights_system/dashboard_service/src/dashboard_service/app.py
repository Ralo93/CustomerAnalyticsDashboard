import hashlib
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
from typing import Dict, Optional, List

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
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", 3600))  # Extended cache time to 5 minutes

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
    

def fetch_data(endpoint: str, cache_key: str, params=None):
    """Fetch data from API with Redis caching"""
    try:
        # Create a cache key that includes the params if any
        full_cache_key = cache_key
        if params:
            param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_cache_key = f"{cache_key}:{param_str}"
        
        if redis_client:
            cached_data = redis_client.get(full_cache_key)
            if cached_data:
                logger.info(f"Retrieved {endpoint} data from Redis cache")
                return pd.DataFrame(json.loads(cached_data))
        
        with httpx.Client(timeout=10.0) as client:
            url = f"{DB_SERVICE_URL}/analytics/{endpoint}"
            if params:
                response = client.get(url, params=params)
            else:
                response = client.get(url)
            
            if response.status_code == 404:
                logger.warning(f"Endpoint {endpoint} not found (404)")
                # Return empty DataFrame with expected columns for each endpoint type
                if endpoint == "sentiment-distribution":
                    return pd.DataFrame(columns=['sales_funnel_stage', 'positive_count', 'negative_count', 'neutral_count', 'total_count'])
                elif endpoint == "impact-scores":
                    return pd.DataFrame(columns=['sales_funnel_stage', 'average_impact', 'count'])
                elif endpoint == "sentiment-impact":
                    return pd.DataFrame(columns=['sentiment', 'business_impact', 'count'])
                elif endpoint == "funnel-metrics":
                    return pd.DataFrame(columns=['stage', 'count', 'percentage'])
                elif endpoint == "intent-distribution":
                    return pd.DataFrame(columns=['intent', 'count', 'percentage'])
                elif endpoint == "high-impact-sentiment":
                    return pd.DataFrame(columns=['sales_funnel_stage', 'positive_high_impact_count', 'total_count', 'percentage'])
                #elif endpoint == "stage-intent-alignment":
                #    return pd.DataFrame(columns=['sales_funnel_stage', 'intent', 'count', 'alignment_score'])
                elif endpoint == "impact-information":
                    return pd.DataFrame(columns=['information_type', 'business_impact', 'count'])
                elif endpoint == "product-mentions":
                    return pd.DataFrame.from_dict({
                        'total_sentences': 0,
                        'with_any_product_mention': 0,
                        'with_multiple_products': 0,
                        'masterblaster': {
                            'mention_count': 0,
                            'percentage': 0,
                            'with_quantity_count': 0,
                            'avg_quantity': 0,
                            'max_quantity': 0
                        },
                        'funpun': {
                            'mention_count': 0,
                            'percentage': 0,
                            'with_quantity_count': 0,
                            'avg_quantity': 0,
                            'max_quantity': 0
                        },
                        'powerpro': {
                            'mention_count': 0,
                            'percentage': 0,
                            'with_quantity_count': 0,
                            'avg_quantity': 0,
                            'max_quantity': 0
                        }
                    }, orient='index').transpose()
                else:
                    return pd.DataFrame()
                
            elif response.status_code != 200:
                error_msg = f"Failed to fetch {endpoint} data: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return pd.DataFrame()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            if redis_client:
                try:
                    redis_client.setex(
                        full_cache_key,
                        CACHE_EXPIRY,  # Use the global constant
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

# Add this function to prepare filterable sentences based on relevance
def prepare_filtered_sentences(all_sentences, relevance_filter="all"):
    """
    Filter sentences based on sales funnel relevance
    
    Args:
        all_sentences: List of sentence objects
        relevance_filter: 'all', 'relevant', or 'non_relevant'
        
    Returns:
        Filtered list of sentences
    """
    if not all_sentences:
        return []
        
    if relevance_filter.lower() == "all":
        return all_sentences
    elif relevance_filter.lower() == "relevant":
        return [s for s in all_sentences if s.get('is_sales_funnel_relevant') == True]
    elif relevance_filter.lower() == "non_relevant":
        return [s for s in all_sentences if s.get('is_sales_funnel_relevant') == False]
    else:
        return all_sentences
    
# Fetch sales funnel relevance statistics
def fetch_relevance_stats():
    """Fetch statistics about sales funnel relevance"""
    try:
        cache_key = "ml_insights:relevance_stats"
        
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Retrieved relevance stats from Redis cache")
                return json.loads(cached_data)
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{DB_SERVICE_URL}/analytics/sales-funnel-relevance")
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch relevance stats: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return {}
            
            data = response.json()
            
            if redis_client:
                try:
                    redis_client.setex(
                        cache_key,
                        CACHE_EXPIRY,
                        json.dumps(data)
                    )
                    logger.info(f"Cached relevance stats for {CACHE_EXPIRY} seconds")
                except Exception as cache_error:
                    logger.error(f"Failed to cache relevance stats: {cache_error}")
            
            return data
    except Exception as e:
        logger.error(f"Error fetching relevance stats: {str(e)}")
        st.error(f"Error fetching relevance stats: {str(e)}")
        return {}

# New function to fetch all sentences at once
def fetch_all_sentences(include_non_relevant=True):
    """Fetch all sentences with their label data for client-side filtering"""
    try:
        cache_key = f"ml_insights:all_sentences:{include_non_relevant}"
        
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Retrieved all sentences from Redis cache")
                return json.loads(cached_data)
        
        with httpx.Client(timeout=20.0) as client:  # Extended timeout for potentially large data
            params = {"include_non_relevant": "true" if include_non_relevant else "false"}
            response = client.get(f"{DB_SERVICE_URL}/analytics/all-sentences", params=params)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch all sentences: {response.status_code}"
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
                    logger.info(f"Cached {len(data)} sentences for {CACHE_EXPIRY} seconds")
                except Exception as cache_error:
                    logger.error(f"Failed to cache sentences: {cache_error}")
            
            return data
    except Exception as e:
        logger.error(f"Error fetching all sentences: {str(e)}")
        st.error(f"Error fetching all sentences: {str(e)}")
        return []

# Update filter_sentences to better handle various filtering cases
def filter_sentences(all_sentences, filters: Dict[str, any], limit: int = 100):
    """Filter sentences client-side based on criteria"""
    if not all_sentences:
        return []
    
    filtered = all_sentences
    
    # Apply filters - handle various data types properly
    for key, value in filters.items():
        if value is not None and value != "All":
            # Boolean filter (for is_sales_funnel_relevant)
            if isinstance(value, bool):
                filtered = [s for s in filtered if s.get(key) == value]
            # String filter (for most other attributes)
            else:
                filtered = [s for s in filtered if str(s.get(key, '')).lower() == str(value).lower()]
    
    # Sort by created_at (newest first) and limit results
    filtered = sorted(filtered, key=lambda x: x.get('created_at', ''), reverse=True)
    return filtered[:limit]

# Modify the display_sentence_details function to better show sales funnel relevance information
def display_sentence_details(sentences_data, limit=100):
    """Display sentence details in an expandable format"""
    if not sentences_data:
        st.info("No sentences match the selected criteria.")
        return
    
    # Only display up to the limit
    display_sentences = sentences_data[:limit]
    
    st.write(f"### Related Sentences ({len(display_sentences)} of {len(sentences_data)} matching)")
    
    for sentence in display_sentences:
        # Create a more descriptive title that shows relevance
        is_relevant = sentence.get('is_sales_funnel_relevant', None)
        relevance_tag = ""
        if is_relevant is True:
            relevance_tag = "🟢 [Sales Funnel Relevant] "
        elif is_relevant is False:
            relevance_tag = "🔴 [Not Sales Funnel Relevant] "
            
        # Add the relevance tag to the title
        with st.expander(f"{relevance_tag}{sentence['text'][:1500]}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Full Text:**")
                st.write(sentence['text'])
                st.write(f"**Created:** {sentence['created_at']}")
            
            with col2:
                st.write("**Attributes:**")
                # Clearly show relevance status
                is_relevant_text = "Yes" if sentence.get('is_sales_funnel_relevant', False) else "No"
                st.write(f"- **Sales Funnel Relevant:** {is_relevant_text}")
                
                # Show confidence score if available
                if 'is_sales_funnel_relevant_confidence' in sentence:
                    confidence = sentence['is_sales_funnel_relevant_confidence']
                    if confidence:
                        st.write(f"- **Relevance Confidence:** {confidence:.2f}")
                
                # Only show stage if relevant
                if sentence.get('is_sales_funnel_relevant', False):
                    sales_funnel_stage = sentence.get('sales_funnel_stage', 'N/A')
                    st.write(f"- **Sales Funnel Stage:** {sales_funnel_stage if sales_funnel_stage else 'N/A'}")
                
                # Always show these attributes for all sentences
                st.write(f"- **Sentiment:** {sentence.get('sentiment', 'N/A')}")
                st.write(f"- **Business Impact:** {sentence.get('business_impact', 'N/A')}")
                st.write(f"- **Intent:** {sentence.get('intent', 'N/A')}")

# New function to render sales funnel relevance statistics
def render_relevance_stats(stats, all_sentences):
    """Display sales funnel relevance statistics"""
    if not stats:
        st.warning("No sales funnel relevance statistics available")
        return
    
    st.subheader("Sales Funnel Relevance Overview")
    
    # Create metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Sentences", 
            f"{stats['total_sentences']:,}"
        )
    
    with col2:
        st.metric(
            "Sales Funnel Relevant", 
            f"{stats['sales_funnel_relevant']:,}", 
            f"{stats['relevant_percentage']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Non-Relevant", 
            f"{stats['non_sales_funnel_relevant']:,}", 
            f"{stats['non_relevant_percentage']:.1f}%"
        )
    
    # Create pie chart data
    pie_data = pd.DataFrame([
        {"category": "Relevant", "count": stats['sales_funnel_relevant']},
        {"category": "Not Relevant", "count": stats['non_sales_funnel_relevant']}
    ])
    
    if stats['unclassified'] > 0:
        pie_data = pie_data.append({"category": "Unclassified", "count": stats['unclassified']}, ignore_index=True)
    
    # Create pie chart
    chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta('count:Q'),
        color=alt.Color('category:N', scale=alt.Scale(
            domain=['Relevant', 'Not Relevant', 'Unclassified'],
            range=['#2ecc71', '#e74c3c', '#95a5a6']
        )),
        tooltip=['category', alt.Tooltip('count:Q', format=',')]
    ).properties(
        width=400,
        height=400,
        title="Distribution of Sales Funnel Relevance"
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Filter selection
    relevant_only = st.checkbox("Show only sales funnel relevant sentences", value=False, key="relevance_stats_checkbox")
    
    if relevant_only:
        filtered_sentences = filter_sentences(all_sentences, {'is_sales_funnel_relevant': True})
    else:
        filtered_sentences = filter_sentences(all_sentences, {})
    
    display_sentence_details(filtered_sentences)

# Visualization functions with client-side filtering
def render_sentiment_distribution(df, all_sentences):
    """Visualize sentiment distribution per sales funnel stage with interactive elements"""
    if df.empty:
        st.warning("No sentiment distribution data available")
        return

    st.subheader("Sentiment Distribution by Sales Funnel Stage")
    st.info("This chart only shows data for sales funnel relevant sentences")
    
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
    
    # Use client-side filtering
    filters = {'is_sales_funnel_relevant': True}  # Always filter for relevant sentences
    if selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    if selected_sentiment != "All":
        filters['sentiment'] = selected_sentiment
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)
def render_impact_scores(df, all_sentences):
    """Visualize average impact scores per stage with enhanced interpretability"""
    if df.empty:
        st.warning("No impact score data available")
        return

    st.subheader("Business Impact Analysis by Sales Funnel Stage")
    
    # Explanation of what impact scores mean
    with st.expander("📊 Understanding Business Impact Scores", expanded=True):
        st.markdown("""
        **What are Business Impact Scores?**
        
        Business impact scores quantify the potential effect of customer sentences on the business:
        
        - **1.0**: Low impact - Minimal effect on business operations or decisions
        - **2.0**: Medium impact - Moderate effect that may warrant attention
        - **3.0**: High impact - Significant effect requiring attention or action
        - **1.5**: Neutral impact - Neither positive nor negative effect
        
        Higher scores indicate areas where customer feedback could have stronger business implications.
        """)
    
    # Create a metrics row to highlight key insights
    col1, col2, col3 = st.columns(3)
    
    # Calculate overall average impact
    overall_avg = (df['average_impact'] * df['count']).sum() / df['count'].sum() if df['count'].sum() > 0 else 0
    
    # Find highest and lowest impact stages
    highest_impact_stage = df.loc[df['average_impact'].idxmax()] if not df.empty else None
    lowest_impact_stage = df.loc[df['average_impact'].idxmin()] if not df.empty else None
    
    with col1:
        st.metric(
            "Overall Average Impact", 
            f"{overall_avg:.2f}", 
            help="Weighted average impact score across all funnel stages"
        )
    
    with col2:
        if highest_impact_stage is not None:
            st.metric(
                "Highest Impact Stage", 
                f"{highest_impact_stage['sales_funnel_stage']}", 
                f"{highest_impact_stage['average_impact']:.2f}",
                help="Stage with the highest average impact score"
            )
    
    with col3:
        if lowest_impact_stage is not None:
            st.metric(
                "Lowest Impact Stage", 
                f"{lowest_impact_stage['sales_funnel_stage']}", 
                f"{lowest_impact_stage['average_impact']:.2f}",
                help="Stage with the lowest average impact score"
            )
    
    # Add a note about the data
    st.info("This analysis only includes sales funnel relevant sentences. Impact scores range from 1.0 (Low) to 3.0 (High).")
    
    # Define a custom color scheme
    color_scale = alt.Scale(
        domain=[1, 1.5, 2, 3],
        range=['#4575b4', '#91bfdb', '#fee090', '#d73027']
    )
    
    # Create enhanced bar chart with reference lines
    base = alt.Chart(df).encode(
        x=alt.X('sales_funnel_stage:N', 
                title='Sales Funnel Stage',
                sort=None),  # Preserve original order
        tooltip=[
            alt.Tooltip('sales_funnel_stage:N', title='Stage'),
            alt.Tooltip('average_impact:Q', format='.2f', title='Avg Impact'),
            alt.Tooltip('count:Q', format=',', title='Sentence Count')
        ]
    )
    
    # Add reference lines for impact levels
    rule_high = alt.Chart(pd.DataFrame({'y': [2.5]})).mark_rule(
        strokeDash=[12, 6],
        strokeWidth=1,
        color='#d73027'
    ).encode(y='y:Q')
    
    rule_medium = alt.Chart(pd.DataFrame({'y': [1.75]})).mark_rule(
        strokeDash=[12, 6],
        strokeWidth=1,
        color='#fee090'
    ).encode(y='y:Q')
    
    # Add text labels for reference lines
    text_high = alt.Chart(pd.DataFrame({'y': [2.5], 'text': ['High Impact Threshold']})).mark_text(
        align='left',
        baseline='bottom',
        dx=5,
        fontSize=11,
        color='#d73027'
    ).encode(y='y:Q', text='text:N')
    
    text_medium = alt.Chart(pd.DataFrame({'y': [1.75], 'text': ['Medium Impact Threshold']})).mark_text(
        align='left',
        baseline='bottom',
        dx=5,
        fontSize=11,
        color='#fee090'
    ).encode(y='y:Q', text='text:N')
    
    # Create the bars with extra encoding
    bars = base.mark_bar().encode(
        y=alt.Y('average_impact:Q', 
                title='Average Business Impact Score',
                scale=alt.Scale(domain=[0.8, 3.2])),  # Adjusted scale to show thresholds
        color=alt.Color('average_impact:Q', 
                        scale=color_scale,
                        legend=alt.Legend(title="Impact Level")),
        size=alt.Size('count:Q', 
                     scale=alt.Scale(range=[30, 60]),
                     legend=alt.Legend(title="Sentence Count"))
    )
    
    # Add counts as text on top of bars
    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        y=alt.Y('average_impact:Q'),
        text=alt.Text('count:Q', format=','),
        color=alt.value('black')
    )
    
    # Combine all chart elements
    chart = (bars + text + rule_high + rule_medium + text_high + text_medium).properties(
        width=600,
        height=400,
        title="Average Business Impact by Sales Funnel Stage"
    ).interactive()

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Business insights based on data
    highest_volume_stage = df.loc[df['count'].idxmax()] if not df.empty else None
    
    if highest_impact_stage is not None and highest_volume_stage is not None:
        st.subheader("Key Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown(f"""
            **High Impact Areas:**
            
            The **{highest_impact_stage['sales_funnel_stage']}** stage shows the highest average impact score 
            ({highest_impact_stage['average_impact']:.2f}), suggesting customer interactions at this stage have 
            significant business implications.
            
            **Recommendation:** Prioritize reviewing and responding to feedback in this stage.
            """)
        
        with insights_col2:
            st.markdown(f"""
            **Volume Considerations:**
            
            The **{highest_volume_stage['sales_funnel_stage']}** stage has the highest volume 
            of interactions ({highest_volume_stage['count']} sentences).
            
            **Recommendation:** Ensure adequate resources are allocated to handle the volume of interactions
            at this stage.
            """)
    
    # Enhanced filtering options
    st.subheader("Explore Sentences by Impact")
    
    # Create two columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        selected_stage = st.selectbox(
            "Select Sales Funnel Stage",
            options=["All"] + sorted(df['sales_funnel_stage'].unique().tolist()),
            key="impact_stage_select"
        )
    
    with filter_col2:
        # Add impact level filter
        impact_levels = {
            "All Levels": None,
            "High Impact (2.5+)": "High",
            "Medium Impact (1.75-2.5)": "Medium",
            "Low Impact (<1.75)": "Low",
            "Neutral Impact": "Neutral"
        }
        
        selected_impact = st.selectbox(
            "Filter by Impact Level",
            options=list(impact_levels.keys()),
            key="impact_level_select"
        )
    
    # Additional filters
    with st.expander("Additional Filters"):
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            options=["All", "Positive", "Negative", "Neutral"],
            key="impact_sentiment_filter"
        )
    
    # Use client-side filtering
    filters = {'is_sales_funnel_relevant': True}  # Always filter for relevant sentences
    
    if selected_stage and selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    if selected_impact != "All Levels":
        filters['business_impact'] = impact_levels[selected_impact]
    
    if sentiment_filter != "All":
        filters['sentiment'] = sentiment_filter
    
    # Filter and display sentences
    filtered_sentences = filter_sentences(all_sentences, filters)
    
    if filtered_sentences:
        st.success(f"Found {len(filtered_sentences)} sentences matching your criteria.")
        display_sentence_details(filtered_sentences)
    else:
        st.info("No sentences found matching your criteria. Try adjusting the filters.")
    
    # Add a contextual note at the bottom
    st.markdown("""
    **Note:** Business impact scores help prioritize which customer interactions need immediate attention. 
    High impact sentences often indicate opportunities for improvement or areas of potential risk.
    """)


def render_sentiment_impact(df, all_sentences):
    """Visualize sentiment and business impact correlation"""
    # Add relevance filter at the top
    relevance_filter = st.radio(
        "Filter by Sales Funnel Relevance",
        ["All Sentences", "Relevant Only", "Non-Relevant Only"],
        horizontal=True,
        key="sentiment_impact_relevance_filter"
    )
    
    # Convert UI selection to API parameter
    api_relevance_param = None
    if relevance_filter == "Relevant Only":
        api_relevance_param = True
    elif relevance_filter == "Non-Relevant Only":
        api_relevance_param = False
    
    # Re-fetch data with the selected filter
    with st.spinner("Updating chart..."):
        # Use the fetch_data function with params
        filtered_df = fetch_data(
            "sentiment-impact", 
            f"ml_insights:sentiment_impact:{relevance_filter}",
            params={"is_sales_funnel_relevant": api_relevance_param} if api_relevance_param is not None else None
        )
    
    if filtered_df.empty:
        st.warning("No sentiment impact data available for the selected filter")
        return

    st.subheader("Sentiment and Business Impact Correlation")
    
    # Check that required columns exist
    if 'sentiment' not in filtered_df.columns or 'business_impact' not in filtered_df.columns:
        st.error("Required columns not found in data")
        return
    
    # Create heatmap using the filtered data
    chart = alt.Chart(filtered_df).mark_rect().encode(
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
    
    # Filter selection controls - using the new filtered dataframe
    available_sentiments = filtered_df['sentiment'].dropna().unique().tolist() if not filtered_df.empty else []
    available_impacts = filtered_df['business_impact'].dropna().unique().tolist() if not filtered_df.empty else []
    
    col1, col2 = st.columns(2)
    with col1:
        selected_sentiment = st.selectbox(
            "Select Sentiment", 
            options=["All"] + sorted(available_sentiments),
            key="sentiment_impact_select"
        )
    
    with col2:
        selected_impact = st.selectbox(
            "Select Business Impact",
            options=["All"] + sorted(available_impacts),
            key="business_impact_select"
        )
    
    # Use client-side filtering for displaying sentences
    filters = {}
    
    # Apply relevance filter
    if relevance_filter == "Relevant Only":
        filters['is_sales_funnel_relevant'] = True
    elif relevance_filter == "Non-Relevant Only":
        filters['is_sales_funnel_relevant'] = False
    
    if selected_sentiment != "All":
        filters['sentiment'] = selected_sentiment
    
    if selected_impact != "All":
        filters['business_impact'] = selected_impact
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)

def render_funnel_metrics(df, all_sentences):
    """Visualize funnel metrics with correct funnel stage order"""
    if df.empty:
        st.warning("No funnel metrics data available")
        return

    st.subheader("Sales Funnel Metrics")
    st.info("This chart only shows data for sales funnel relevant sentences")
    
    # Define the correct order for funnel stages
    funnel_order = ["Awareness", "Interest", "Consideration", "Intent", "Evaluation", "Purchase"]
    
    # Make a copy to avoid modifying original dataframe
    ordered_df = df.copy()
    
    # Create a categorical type with our custom order
    ordered_df['stage'] = pd.Categorical(
        ordered_df['stage'],
        categories=funnel_order,
        ordered=True
    )
    
    # Sort by the ordered categorical
    ordered_df = ordered_df.sort_values('stage')
    
    # Create funnel chart
    chart = alt.Chart(ordered_df).mark_bar().encode(
        x=alt.X('count:Q', title='Count'),
        y=alt.Y('stage:N', 
                title='Funnel Stage',
                sort=funnel_order),  # Explicitly set sort order
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
        options=["All"] + funnel_order,  # Use the same order for consistency
        key="funnel_stage_select"
    )
    
    # Always filter for relevant sentences since this is funnel data
    filters = {'is_sales_funnel_relevant': True}
    if selected_stage and selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)

# Update render_intent_distribution with improved relevance filtering
def render_intent_distribution(df, all_sentences):
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
    
    # Add relevance filter with a more intuitive radio button
    #relevance_filter = st.radio(
    #    "Filter by Sales Funnel Relevance",
    #    ["All Sentences", "Relevant Only", "Non-Relevant Only"],
    #    horizontal=True,
    #    key="intent_relevance_filter"
    #)
    
    # Filter selection
    selected_intent = st.selectbox(
        "Select Intent to View Sentences",
        options=["All"] + sorted(df['intent'].unique().tolist()),
        key="intent_select"
    )
    
    # Use client-side filtering
    filters = {}
    
    # Apply relevance filter
    #if relevance_filter == "Relevant Only":
    filters['is_sales_funnel_relevant'] = False
    #elif relevance_filter == "Non-Relevant Only":
    #    filters['is_sales_funnel_relevant'] = False
    
    if selected_intent and selected_intent != "All":
        filters['intent'] = selected_intent
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)

def render_high_impact_sentiment(df, all_sentences):
    """
    Visualize high and critical business impact sentences along with their sentiment distribution.
    Oriented similarly to the Sentiment Distribution by Sales Funnel Stage visualization.
    """
    st.subheader("High Business Impact Analysis")
    
    # Create a filter for impact level
    impact_filter = st.radio(
        "Select Business Impact Level",
        ["High & Critical", "High Only", "Critical Only"],
        horizontal=True,
        key="impact_level_filter"
    )
    
    # Fetch data based on the selected impact levels
    with st.spinner("Loading high impact data..."):
        if impact_filter == "High Only":
            impact_params = {"business_impact": "High"}
            impact_title = "High"
        elif impact_filter == "Critical Only":
            impact_params = {"business_impact": "Critical"}
            impact_title = "Critical"
        else:  # "High & Critical"
            impact_params = {"business_impact": ["High", "Critical"]}
            impact_title = "High & Critical"
        
        # Fetch the filtered high impact data
        high_impact_data = fetch_data(
            "sentiment-impact", 
            f"ml_insights:high_impact_{impact_filter.lower().replace(' ', '_')}",
            params=impact_params
        )
    
    if high_impact_data.empty:
        st.warning(f"No {impact_title} business impact data available")
        return
    
    # Create a chart showing sentiment distribution for high impact sentences
    st.subheader(f"{impact_title} Business Impact Sentiment Distribution")
    
    # Prepare data for visualization
    # If data doesn't have sales_funnel_stage column, add a filter to allow showing by stage
    show_by_stage = False
    if 'sales_funnel_stage' in high_impact_data.columns:
        show_by_stage = st.checkbox("Show distribution by sales funnel stage", value=True)
    
    if show_by_stage and 'sales_funnel_stage' in high_impact_data.columns:
        # Chart showing distribution by stage, similar to sentiment distribution
        chart = alt.Chart(high_impact_data).mark_bar().encode(
            x=alt.X('sales_funnel_stage:N', title='Sales Funnel Stage'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('sentiment:N', scale=alt.Scale(
                domain=['Positive', 'Negative', 'Neutral'],
                range=['#2ecc71', '#e74c3c', '#95a5a6']
            )),
            tooltip=[
                alt.Tooltip('sales_funnel_stage:N', title='Stage'),
                alt.Tooltip('sentiment:N', title='Sentiment'),
                alt.Tooltip('count:Q', format=',', title='Count'),
                alt.Tooltip('business_impact:N', title='Business Impact')
            ]
        ).properties(
            width=600,
            height=400
        ).interactive()
    else:
        # Donut chart for overall sentiment distribution
        chart = alt.Chart(high_impact_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta('count:Q'),
            color=alt.Color('sentiment:N', scale=alt.Scale(
                domain=['Positive', 'Negative', 'Neutral'],
                range=['#2ecc71', '#e74c3c', '#95a5a6']
            )),
            tooltip=[
                alt.Tooltip('sentiment:N', title='Sentiment'),
                alt.Tooltip('count:Q', format=',', title='Count'),
                alt.Tooltip('business_impact:N', title='Business Impact')
            ]
        ).properties(
            width=500,
            height=500,
            title=f"{impact_title} Business Impact Sentiment Distribution"
        ).interactive()
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Add metrics row to show totals
    col1, col2, col3 = st.columns(3)
    
    # Calculate totals by sentiment
    total_count = high_impact_data['count'].sum()
    
    # Group by sentiment to get counts
    sentiment_counts = high_impact_data.groupby('sentiment')['count'].sum().to_dict()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    # Calculate percentages
    positive_pct = (positive_count / total_count * 100) if total_count > 0 else 0
    negative_pct = (negative_count / total_count * 100) if total_count > 0 else 0
    neutral_pct = (neutral_count / total_count * 100) if total_count > 0 else 0
    
    with col1:
        st.metric(
            "Positive Sentiment", 
            f"{positive_count:,}", 
            f"{positive_pct:.1f}%"
        )
    
    with col2:
        st.metric(
            "Negative Sentiment", 
            f"{negative_count:,}", 
            f"{negative_pct:.1f}%"
        )
    
    with col3:
        st.metric(
            "Neutral Sentiment", 
            f"{neutral_count:,}", 
            f"{neutral_pct:.1f}%"
        )
    
    # Filter selection controls
    col1, col2 = st.columns(2)
    with col1:
        if show_by_stage and 'sales_funnel_stage' in high_impact_data.columns:
            selected_stage = st.selectbox(
                "Select Sales Funnel Stage", 
                options=["All"] + sorted(high_impact_data['sales_funnel_stage'].unique().tolist()),
                key="high_impact_stage_filter"
            )
        else:
            selected_stage = "All"
    
    with col2:
        selected_sentiment = st.selectbox(
            "Select Sentiment",
            options=["All", "Positive", "Negative", "Neutral"],
            key="high_impact_sentiment_filter"
        )
    
    # Use client-side filtering
    filters = {}
    
    # Apply impact filter
    if impact_filter == "High Only":
        filters['business_impact'] = "High"
    elif impact_filter == "Critical Only":
        filters['business_impact'] = "Critical"
    else:
        # For "High & Critical", we need to handle this specially since our filter function
        # doesn't support OR conditions directly
        # We'll retrieve both sets and combine them
        high_filters = filters.copy()
        high_filters['business_impact'] = "High"
        critical_filters = filters.copy()
        critical_filters['business_impact'] = "Critical"
        
        high_sentences = filter_sentences(all_sentences, high_filters)
        critical_sentences = filter_sentences(all_sentences, critical_filters)
        
        # Combine both sets (avoiding duplicates if any)
        combined_sentences = high_sentences + [s for s in critical_sentences if s not in high_sentences]
        
        # Apply additional filters
        if selected_stage != "All":
            combined_sentences = [s for s in combined_sentences if s.get('sales_funnel_stage') == selected_stage]
        
        if selected_sentiment != "All":
            combined_sentences = [s for s in combined_sentences if s.get('sentiment') == selected_sentiment]
        
        display_sentence_details(combined_sentences)
        return
    
    # Continue with standard filtering for "High Only" or "Critical Only" cases
    if selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    if selected_sentiment != "All":
        filters['sentiment'] = selected_sentiment
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)

    # Additional analysis - Top issues or patterns
    if not high_impact_data.empty:
        st.subheader(f"Analysis of {impact_title} Business Impact Sentences")
        
        # If we have other attributes like 'information_type' or 'intent', show their distribution
        if 'information_type' in high_impact_data.columns:
            # Show information type distribution
            info_chart = alt.Chart(high_impact_data).mark_bar().encode(
                x=alt.X('information_type:N', title='Information Type'),
                y=alt.Y('count:Q', title='Count'),
                color=alt.Color('information_type:N', scale=alt.Scale(scheme='category10')),
                tooltip=['information_type', 'count']
            ).properties(
                width=600,
                height=300,
                title=f"Information Types in {impact_title} Business Impact Sentences"
            ).interactive()
            
            st.altair_chart(info_chart, use_container_width=True)
        
        if 'intent' in high_impact_data.columns:
            # Show intent distribution
            intent_chart = alt.Chart(high_impact_data).mark_bar().encode(
                x=alt.X('intent:N', title='Intent'),
                y=alt.Y('count:Q', title='Count'),
                color=alt.Color('intent:N', scale=alt.Scale(scheme='category10')),
                tooltip=['intent', 'count']
            ).properties(
                width=600,
                height=300,
                title=f"Intent Distribution in {impact_title} Business Impact Sentences"
            ).interactive()
            
            st.altair_chart(intent_chart, use_container_width=True)

def render_stage_intent_alignment(df, all_sentences):
    """Visualize alignment between sales funnel stages and intents"""
    if df.empty:
        st.warning("No stage-intent alignment data available")
        return

    st.subheader("Sales Funnel Stage and Intent Alignment")
    st.info("This chart only shows data for sales funnel relevant sentences")
    
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
    
    # Use client-side filtering
    filters = {'is_sales_funnel_relevant': True}  # Always filter for relevant sentences
    if selected_stage != "All":
        filters['sales_funnel_stage'] = selected_stage
    
    if selected_intent != "All":
        filters['intent'] = selected_intent
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)

def render_impact_information(df, all_sentences):
    """Visualize relationship between information type and business impact"""
    # Add relevance filter at the top
    relevance_filter = st.radio(
        "Filter by Sales Funnel Relevance",
        ["All Sentences", "Relevant Only", "Non-Relevant Only"],
        horizontal=True,
        key="impact_info_relevance_filter"
    )
    
    # Convert UI selection to API parameter
    api_relevance_param = None
    if relevance_filter == "Relevant Only":
        api_relevance_param = True
    elif relevance_filter == "Non-Relevant Only":
        api_relevance_param = False
    
    # Re-fetch data with the selected filter
    with st.spinner("Updating chart..."):
        # Use the fetch_data function with params
        filtered_df = fetch_data(
            "impact-information", 
            "ml_insights:impact_info",
            params={"is_sales_funnel_relevant": api_relevance_param} if api_relevance_param is not None else None
        )
    
    if filtered_df.empty:
        st.warning("No impact-information data available for the selected filter")
        return
    
    st.subheader("Information Type and Business Impact Relationship")
    
    # Create grouped bar chart with the filtered data
    chart = alt.Chart(filtered_df).mark_bar().encode(
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
            options=["All"] + sorted(filtered_df['information_type'].unique().tolist()),
            key="info_type_select"
        )
    
    with col2:
        selected_impact = st.selectbox(
            "Select Business Impact",
            options=["All"] + sorted(filtered_df['business_impact'].unique().tolist()),
            key="info_impact_select"
        )
    
    # Use client-side filtering for the sentence display
    filters = {}
    
    # Apply relevance filter
    if relevance_filter == "Relevant Only":
        filters['is_sales_funnel_relevant'] = True
    elif relevance_filter == "Non-Relevant Only":
        filters['is_sales_funnel_relevant'] = False
    
    if selected_info_type != "All":
        filters['intent'] = selected_info_type
    
    if selected_impact != "All":
        filters['business_impact'] = selected_impact
    
    filtered_sentences = filter_sentences(all_sentences, filters)
    display_sentence_details(filtered_sentences)

def render_product_mentions(df, all_sentences):
    """Visualize product mention statistics with enhanced filtering"""
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Log the raw input data
    logger.debug(f"Raw input data type: {type(df)}")
    logger.debug(f"Raw input data contents: {df}")

    # Validate and normalize input data
    if df is None or df.empty:
        logger.error("No data provided for product mentions")
        st.error("No data provided for product mentions")
        return

    # Extract data based on the DataFrame structure shown in the logs
    # The DataFrame appears to have rows for mention_count, quantity, percentage
    # and columns for each product

    # First, check if we have the expected structure
    try:
        # Check if the DataFrame has the expected index (rows)
        if isinstance(df.index, pd.Index) and 'mention_count' in df.index and 'quantity' in df.index:
            # This is the structure we see in the logs - extract accordingly
            logger.debug("Found transposed DataFrame structure with metrics as rows")
            
            # Extract total values
            total_sentences = df.loc['mention_count', 'total_sentences']
            with_any_product = df.loc['mention_count', 'with_any_product_mention']
            with_multiple = df.loc['mention_count', 'with_multiple_products']
            
            # Product columns
            product_keys = ['masterblaster', 'funpun', 'powerpro']
            product_names = ['MasterBlaster', 'FunPun', 'PowerPro']
            
            # Prepare product data
            product_data = []
            
            for product_key, product_name in zip(product_keys, product_names):
                if product_key in df.columns:
                    mention_count = df.loc['mention_count', product_key]
                    quantity = df.loc['quantity', product_key]
                    percentage = df.loc['percentage', product_key]
                    
                    logger.debug(f"Product {product_name} - Extracted data: Mentions={mention_count}, Quantity={quantity}, Percentage={percentage}")
                    
                    # Calculate average quantity if we have mentions
                    avg_quantity = quantity / mention_count if mention_count > 0 else 0
                    
                    # Calculate a reasonable max quantity estimate
                    # If we have only 1 mention, max = total
                    # If we have multiple mentions, estimate max as higher than average
                    if mention_count <= 1:
                        max_quantity = quantity  # If only one mention, max = total
                    else:
                        # Estimate max as approximately 2-3x the average
                        # This is a heuristic since we don't have individual values
                        max_quantity = min(quantity, avg_quantity * 2.5)  # Cap at total quantity
                    
                    product_data.append({
                        'Product': product_name,
                        'Mention Count': float(mention_count),
                        'Percentage': float(percentage),
                        'With Quantity Count': float(mention_count) if quantity > 0 else 0,
                        'Avg Quantity': float(avg_quantity),
                        'Max Quantity': float(max_quantity),
                        'Total Quantity': float(quantity),
                        'Percentage Display': f"{percentage:.1f}%"
                    })
                else:
                    logger.warning(f"Product column {product_key} not found in DataFrame")
        else:
            # Try to handle it as a regular DataFrame or dict
            logger.debug("Trying to handle as standard DataFrame or dict")
            if isinstance(df, pd.DataFrame):
                df = df.to_dict('records')[0] if len(df) > 0 else {}
            
            # Extract from dictionary structure
            total_sentences = df.get('total_sentences', 0)
            with_any_product = df.get('with_any_product_mention', 0)
            with_multiple = df.get('with_multiple_products', 0)
            
            product_keys = ['masterblaster', 'funpun', 'powerpro']
            product_names = ['MasterBlaster', 'FunPun', 'PowerPro']
            
            product_data = []
            
            for product_key, product_name in zip(product_keys, product_names):
                product_info = df.get(product_key, {})
                
                if isinstance(product_info, dict):
                    mention_count = product_info.get('mention_count', 0)
                    quantity = product_info.get('quantity', 0)
                    percentage = product_info.get('percentage', 0)
                else:
                    mention_count = product_info
                    quantity = df.get(f'{product_key}_quantity', 0)
                    percentage = df.get(f'{product_key}_percentage', 0)
                
                logger.debug(f"Product {product_name} - Dict data: Mentions={mention_count}, Quantity={quantity}, Percentage={percentage}")
                
                # Calculate average quantity
                avg_quantity = quantity / mention_count if mention_count > 0 else 0
                
                # Calculate max quantity
                if mention_count <= 1:
                    max_quantity = quantity
                else:
                    max_quantity = min(quantity, avg_quantity * 2.5)
                
                product_data.append({
                    'Product': product_name,
                    'Mention Count': float(mention_count),
                    'Percentage': float(percentage),
                    'With Quantity Count': float(mention_count) if quantity > 0 else 0,
                    'Avg Quantity': float(avg_quantity),
                    'Max Quantity': float(max_quantity),
                    'Total Quantity': float(quantity),
                    'Percentage Display': f"{percentage:.1f}%"
                })
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        st.error(f"Could not process product mention data: {e}")
        return

    # Log the extracted values
    logger.debug(f"Total Sentences: {total_sentences}")
    logger.debug(f"With Any Product: {with_any_product}")
    logger.debug(f"With Multiple Products: {with_multiple}")
    
    # Create DataFrame from product data
    try:
        product_df = pd.DataFrame(product_data)
        
        # Log the final DataFrame
        logger.debug("Final Product DataFrame:")
        logger.debug(product_df.to_string())
        
        # Explicitly set column types to avoid Altair type inference issues
        product_df = product_df.astype({
            'Product': 'category',
            'Mention Count': 'float64',
            'Percentage': 'float64',
            'With Quantity Count': 'float64',
            'Avg Quantity': 'float64',
            'Max Quantity': 'float64',
            'Total Quantity': 'float64',
            'Percentage Display': 'string'
        })
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        st.error(f"Could not create visualization DataFrame: {e}")
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sentences", f"{total_sentences:,}")
    
    with col2:
        percentage_with_product = (with_any_product / total_sentences * 100) if total_sentences > 0 else 0
        st.metric("Sentences with Products", f"{with_any_product:,}", f"{percentage_with_product:.1f}%")
    
    #with col3:
    #    st.metric("Multiple Product Mentions", f"{with_multiple:,}")
    
    # Visualization
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            mention_chart = alt.Chart(product_df).mark_bar().encode(
                x=alt.X('Product:N', title='Product', sort=product_names),
                y=alt.Y('Mention Count:Q', title='Number of Mentions', scale=alt.Scale(zero=True)),
                color=alt.Color('Product:N', scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('Product:N'),
                    alt.Tooltip('Mention Count:Q', title='Mentions'),
                    alt.Tooltip('With Quantity Count:Q', title='Mentions with Quantity'),
                    alt.Tooltip('Avg Quantity:Q', title='Average Quantity', format='.2f'),
                    alt.Tooltip('Max Quantity:Q', title='Max Quantity', format='.0f'),
                    alt.Tooltip('Total Quantity:Q', title='Total Quantity', format='.0f'),
                    alt.Tooltip('Percentage Display:N', title='Percentage of Total Sentences')
                ]
            ).properties(
                width=300,
                height=400,
                title='Product Mentions Distribution'
            ).interactive()
            
            st.altair_chart(mention_chart, use_container_width=True)
        
        with col2:
            percentage_chart = alt.Chart(product_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('Mention Count:Q', stack=True),
                color=alt.Color('Product:N', scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('Product:N'),
                    alt.Tooltip('Mention Count:Q', title='Mentions'),
                    alt.Tooltip('With Quantity Count:Q', title='Mentions with Quantity'),
                    alt.Tooltip('Avg Quantity:Q', title='Average Quantity', format='.2f'),
                    alt.Tooltip('Max Quantity:Q', title='Max Quantity', format='.0f'),
                    alt.Tooltip('Total Quantity:Q', title='Total Quantity', format='.0f'),
                    alt.Tooltip('Percentage Display:N', title='Percentage of Total Sentences')
                ]
            ).properties(
                width=300,
                height=400,
                title='Product Mentions Percentage'
            ).interactive()
            
            st.altair_chart(percentage_chart, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        st.error(f"Could not create visualizations: {e}")
        return

    # Quantity details with logging
    st.subheader("Quantity Details")
    
    # Create a quantity details dataframe with clearer representations
    quantity_df = pd.DataFrame([
        {"Metric": "Total Quantity", "MasterBlaster": product_df.loc[0, "Total Quantity"], "FunPun": product_df.loc[1, "Total Quantity"], "PowerPro": product_df.loc[2, "Total Quantity"]},
        {"Metric": "Average Quantity", "MasterBlaster": product_df.loc[0, "Avg Quantity"], "FunPun": product_df.loc[1, "Avg Quantity"], "PowerPro": product_df.loc[2, "Avg Quantity"]},
        {"Metric": "Maximum Quantity", "MasterBlaster": product_df.loc[0, "Max Quantity"], "FunPun": product_df.loc[1, "Max Quantity"], "PowerPro": product_df.loc[2, "Max Quantity"]}
    ])
    
    # Format the quantity values
    for col in ["MasterBlaster", "FunPun", "PowerPro"]:
        quantity_df[col] = quantity_df.apply(
            lambda row: f"{row[col]:.2f}" if row["Metric"] == "Average Quantity" else f"{row[col]:.0f}", 
            axis=1
        )
    
    # Log and display the quantity details
    logger.debug("Quantity Details DataFrame:")
    logger.debug(quantity_df.to_string())
    st.dataframe(quantity_df)
    
    # Add overall summary metrics
    if product_df['Total Quantity'].sum() > 0:
        st.subheader("Quantity Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_qty = product_df['Total Quantity'].sum()
            st.metric("Total Overall Quantity", f"{total_qty:.0f}")
        
        with col2:
            # Calculate overall average across products with quantities
            avg_qty = product_df.loc[product_df['Mention Count'] > 0, 'Avg Quantity'].mean()
            st.metric("Overall Average Quantity", f"{avg_qty:.2f}")
        
        with col3:
            # Get overall maximum
            max_qty = product_df['Max Quantity'].max()
            st.metric("Overall Maximum Quantity", f"{max_qty:.0f}")
    
    # NEW SECTION FOR SPECIALIZED FILTERING
    st.markdown("---")
    st.subheader("Product Sentence Finder")
    st.markdown("Use the filters below to find sentences mentioning specific products.")
    
    # Create filter controls in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        selected_product = st.selectbox(
            "Filter by Product",
            options=["All", "MasterBlaster", "FunPun", "PowerPro"],
            key="product_filter_selectbox"
        )
    
    with col2:
        min_quantity = st.number_input(
            "Minimum Quantity",
            min_value=0,
            value=0,
            key="product_min_quantity"
        )
    
    # Create additional filter options
    with st.expander("Additional Filters"):
        relevance_filter = st.radio(
            "Filter by Sales Funnel Relevance",
            ["All", "Relevant Only", "Non-Relevant Only"],
            horizontal=True,
            key="product_relevance_filter"
        )
        
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            options=["All", "Positive", "Negative", "Neutral"],
            key="product_sentiment_filter"
        )
        
        impact_filter = st.selectbox(
            "Filter by Business Impact",
            options=["All", "High", "Medium", "Low", "Neutral"],
            key="product_impact_filter"
        )
    
    # Apply filter button
    filter_clicked = st.button("Find Matching Sentences")
    
    if filter_clicked:
        # Prepare filter parameters for the API call
        params = {}
        
        # Product filters
        if selected_product != "All":
            product_key = f"mentions_{selected_product.lower()}"
            params[product_key] = True
            
            if min_quantity > 0:
                params[f"min_{selected_product.lower()}_quantity"] = min_quantity
        
        # Relevance filter
        if relevance_filter == "Relevant Only":
            params["is_sales_funnel_relevant"] = True
        elif relevance_filter == "Non-Relevant Only":
            params["is_sales_funnel_relevant"] = False
        
        # Sentiment filter
        if sentiment_filter != "All":
            params["sentiment"] = sentiment_filter
            
        # Business impact filter
        if impact_filter != "All":
            params["business_impact"] = impact_filter
        
        # Show loading indicator
        with st.spinner("Fetching sentences..."):
            # Create a cache key based on the parameters
            param_str = json.dumps(params, sort_keys=True)
            cache_key = f"ml_insights:product_sentences:{hashlib.md5(param_str.encode()).hexdigest()}"
            
            # Check if we have this in cache first
            filtered_sentences = None
            if redis_client:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    logger.info("Retrieved filtered product sentences from Redis cache")
                    filtered_sentences = json.loads(cached_data)
            
            # If not in cache, fetch from API
            if not filtered_sentences:
                try:
                    with httpx.Client(timeout=10.0) as client:
                        response = client.get(f"{DB_SERVICE_URL}/analytics/sentences-by-product-filter", params=params)
                        
                        if response.status_code != 200:
                            logger.error(f"Failed to fetch filtered sentences: {response.status_code}")
                            st.error("Failed to fetch sentences. Please try again.")
                            return
                        
                        filtered_sentences = response.json()
                        
                        # Cache the results
                        if redis_client:
                            try:
                                redis_client.setex(
                                    cache_key,
                                    CACHE_EXPIRY,
                                    json.dumps(filtered_sentences)
                                )
                                logger.info(f"Cached filtered product sentences for {CACHE_EXPIRY} seconds")
                            except Exception as cache_error:
                                logger.error(f"Failed to cache filtered sentences: {cache_error}")
                                
                except Exception as e:
                    logger.error(f"Error fetching filtered sentences: {e}")
                    st.error(f"An error occurred: {str(e)}")
                    return
            
            # Display the filtered sentences
            if not filtered_sentences:
                st.info("No sentences found matching your filter criteria.")
            else:
                st.success(f"Found {len(filtered_sentences)} sentences matching your filter criteria.")
                display_sentence_details(filtered_sentences)
    else:
        # If no search performed yet, show a prompt
        st.info("Use the filters above and click 'Find Matching Sentences' to see product-related sentences.")
        
    # Display a note about the filtering
    st.markdown("""
    **Note:** This search uses advanced product-specific filtering from the sentence features table, 
    allowing you to find sentences based on the products they mention and quantities.
    """)
def render_trends_tab():
    """Render the Time Series Trends tab (tab9) with focus on trend visualization"""
    #st.subheader("Time Series Trends Analysis")
    #st.markdown("""
    #This tab analyzes customer interaction trends over time, using external_id as a sequence number.
    #""")
    
    # Initialize session state for period selection if not already set
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = 'all'
    
    # Fetch the external ID range
    try:
        # Check cache first
        id_range = None
        cache_key = "ml_insights:external_id_range"
        
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Retrieved external ID range from Redis cache")
                id_range = json.loads(cached_data)
        
        # If not in cache, fetch from API
        if not id_range:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{DB_SERVICE_URL}/analytics/external-id-range")
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch external ID range: {response.status_code}")
                    st.error("Failed to fetch external ID range. Please try again later.")
                    return
                
                id_range = response.json()
                
                # Cache the result
                if redis_client:
                    try:
                        redis_client.setex(
                            cache_key,
                            CACHE_EXPIRY,
                            json.dumps(id_range)
                        )
                        logger.info(f"Cached external ID range for {CACHE_EXPIRY} seconds")
                    except Exception as cache_error:
                        logger.error(f"Failed to cache external ID range: {cache_error}")
        
        min_id = id_range.get('min_id', 0)
        max_id = id_range.get('max_id', 0)
        count = id_range.get('count', 0)
        
    except Exception as e:
        logger.error(f"Error fetching external ID range: {str(e)}")
        st.error("Failed to load data range. Please try again later.")
        return
    
    # Create more compact filter buttons
    st.markdown("### Select Time Period")
    
    # Use a container with custom CSS for more compact buttons
    filter_container = st.container()
    
    # Create a more compact button layout
    button_cols = filter_container.columns([1, 1, 1, 3])  # The last column is for spacing
    
    # Define a callback for when a period button is clicked
    def on_period_click(period):
        st.session_state.selected_period = period
        st.session_state.period_changed = True
    
    # Create compact, adjacent buttons
    with button_cols[0]:
        last7_selected = st.session_state.selected_period == 'last7'
        st.button(
            "7 days", 
            key="btn_last7", 
            type="primary" if last7_selected else "secondary",
            on_click=on_period_click,
            args=('last7',),
            use_container_width=True
        )
    
    with button_cols[1]:
        last30_selected = st.session_state.selected_period == 'last30'
        st.button(
            "30 days", 
            key="btn_last30", 
            type="primary" if last30_selected else "secondary",
            on_click=on_period_click,
            args=('last30',),
            use_container_width=True
        )
    
    with button_cols[2]:
        all_selected = st.session_state.selected_period == 'all'
        st.button(
            "All time", 
            key="btn_all", 
            type="primary" if all_selected else "secondary",
            on_click=on_period_click,
            args=('all',),
            use_container_width=True
        )
    
    # Calculate ID range based on selection
    start_id = None
    if st.session_state.selected_period == 'last7':
        start_id = max(min_id, max_id - 6)
    elif st.session_state.selected_period == 'last30':
        start_id = max(min_id, max_id - 29)
    else:
        # For 'all', use the full range
        start_id = min_id
    
    # Make sure end_id is always max_id
    end_id = max_id
    
    # Display selection
    if st.session_state.selected_period == 'last7':
        time_range_text = f"Analyzing last 7 interactions (ID {start_id} to {end_id})"
    elif st.session_state.selected_period == 'last30':
        time_range_text = f"Analyzing last 30 interactions (ID {start_id} to {end_id})"
    else:
        time_range_text = f"Analyzing all interactions (ID {min_id} to {end_id})"
    
    st.markdown(f"**{time_range_text}**")
    
    # Add a refresh button for data reload
    st.button("🔄 Refresh Data", key="refresh_trends", help="Reload the trend data from the server")
    
    # Fetch time series trends data
    cache_key = f"ml_insights:time_series:{st.session_state.selected_period}"
    
    # Check if we should skip cache (when refresh is clicked)
    skip_cache = False
    if st.session_state.get('period_changed', False) or st.session_state.get('refresh_clicked', False):
        skip_cache = True
        # Reset states after use
        st.session_state.period_changed = False
        st.session_state.refresh_clicked = False
    
    # Check cache first (unless we're forcing a refresh)
    trend_data = None
    if redis_client and not skip_cache:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Retrieved time series trends from cache for {st.session_state.selected_period}")
            trend_data = json.loads(cached_data)
    
    # If not in cache, fetch from API
    if not trend_data:
        with st.spinner("Loading trend data..."):
            try:
                with httpx.Client(timeout=15.0) as client:
                    params = {}
                    if start_id is not None:
                        params["start_external_id"] = start_id
                    if end_id is not None:
                        params["end_external_id"] = end_id
                    
                    response = client.get(f"{DB_SERVICE_URL}/analytics/time-series-trends", params=params)
                    
                    if response.status_code != 200:
                        logger.error(f"Failed to fetch time series trends: {response.status_code}")
                        st.error("Failed to load trend data. Please try again later.")
                        return
                    
                    trend_data = response.json()
                    
                    # Cache the result
                    if redis_client:
                        try:
                            redis_client.setex(
                                cache_key,
                                CACHE_EXPIRY,
                                json.dumps(trend_data)
                            )
                            logger.info(f"Cached time series trends for {CACHE_EXPIRY} seconds")
                        except Exception as cache_error:
                            logger.error(f"Failed to cache time series trends: {cache_error}")
            except Exception as e:
                logger.error(f"Error fetching time series trends: {str(e)}")
                st.error("Failed to load trend data. Please try again later.")
                return
    
    # Check if we have data to display
    if not trend_data or not trend_data.get('time_points'):
        st.warning("No trend data available for the selected period.")
        return
    
    # Extract data for visualization
    time_points = trend_data.get('time_points', [])
    
    # Make sure time_points are properly formatted for display
    # Convert numbers to strings for display
    formatted_time_points = [str(point) for point in time_points]
    
    # Create tabs for different trend visualizations
    trend_tabs = st.tabs([
        "Sentiment Trends", 
        "Product Mention Trends", 
        "Business Impact Trends",
        "Sales Funnel Trends"
    ])
    
    # 1. Sentiment Trends Tab
    with trend_tabs[0]:
        #st.subheader("Sentiment Trends Over Time")
        
        # Prepare data for chart - with more focus on trends
        sentiment_data = []
        for i, point in enumerate(formatted_time_points):
            sentiment_counts = trend_data.get('sentiment_trends', [])[i]
            total = sentiment_counts.get('total', 1)  # Avoid division by zero
            
            sentiment_data.append({
                'time_point': point,
                'time_index': i,  # Add index for trend line
                'Positive': sentiment_counts.get('positive', 0),
                'Negative': sentiment_counts.get('negative', 0),
                'Neutral': sentiment_counts.get('neutral', 0),
                'Positive %': round(sentiment_counts.get('positive', 0) / total * 100, 1) if total > 0 else 0,
                'Negative %': round(sentiment_counts.get('negative', 0) / total * 100, 1) if total > 0 else 0,
                'Neutral %': round(sentiment_counts.get('neutral', 0) / total * 100, 1) if total > 0 else 0
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Calculate summary statistics for sentiment
        avg_positive = sentiment_df['Positive %'].mean()
        avg_negative = sentiment_df['Negative %'].mean()
        avg_neutral = sentiment_df['Neutral %'].mean()
        
        # Calculate total counts
        total_positive = sentiment_df['Positive'].sum()
        total_negative = sentiment_df['Negative'].sum()
        total_neutral = sentiment_df['Neutral'].sum()
        
        # Calculate positive to negative ratio
        if total_negative > 0:
            pos_neg_ratio = total_positive / total_negative
        else:
            pos_neg_ratio = total_positive if total_positive > 0 else 0
        
        # Display sentiment summary statistics
        st.markdown("#### Summary Statistics")
        
        # Use columns for a clean layout
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("Avg Positive", f"{avg_positive:.1f}%")
        
        with stat_cols[1]:
            st.metric("Avg Negative", f"{avg_negative:.1f}%")
        
        with stat_cols[2]:
            st.metric("Avg Neutral", f"{avg_neutral:.1f}%")
        
        with stat_cols[3]:
            st.metric("Pos/Neg Ratio", f"{pos_neg_ratio:.2f}")
        
        # Reshape for percentage chart (focusing on trends)
        sentiment_pct_df = pd.melt(
            sentiment_df,
            id_vars=['time_point', 'time_index'],
            value_vars=['Positive %', 'Negative %', 'Neutral %'],
            var_name='Sentiment',
            value_name='Percentage'
        )
        
        # Create a larger, more prominent trend visualization
        sentiment_chart = alt.Chart(sentiment_pct_df).mark_line(
            point=True,
            strokeWidth=3,  # Thicker lines for better visibility
        ).encode(
            x=alt.X('time_point:N', 
                   title='Time (External ID)', 
                   sort=None,
                   axis=alt.Axis(labelAngle=-45)),  # Angled labels for better readability
            y=alt.Y('Percentage:Q', 
                   title='Percentage', 
                   scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Sentiment:N', 
                          legend=alt.Legend(title="Sentiment"),
                          scale=alt.Scale(
                              domain=['Positive %', 'Negative %', 'Neutral %'],
                              range=['#2ecc71', '#e74c3c', '#95a5a6']
                          )),
            tooltip=['time_point:N', 'Sentiment:N', 'Percentage:Q']
        ).properties(
            width='container',
            height=400  # Taller chart for better trend visibility
        ).interactive()
        
        # Display the chart
        st.altair_chart(sentiment_chart, use_container_width=True)
    
    # 2. Product Mention Trends Tab
    with trend_tabs[1]:
        #st.subheader("Product Mention Trends Over Time")
        
        # Prepare data with focus on trends
        product_data = []
        for i, point in enumerate(formatted_time_points):
            product_counts = trend_data.get('product_trends', [])[i]
            total = product_counts.get('total', 1)  # Avoid division by zero
            
            product_data.append({
                'time_point': point,
                'time_index': i,  # Add index for trend line
                'MasterBlaster': product_counts.get('masterblaster', 0),
                'FunPun': product_counts.get('funpun', 0),
                'PowerPro': product_counts.get('powerpro', 0),
                'Any Product': product_counts.get('any_product', 0),
                'MasterBlaster %': round(product_counts.get('masterblaster', 0) / total * 100, 1) if total > 0 else 0,
                'FunPun %': round(product_counts.get('funpun', 0) / total * 100, 1) if total > 0 else 0,
                'PowerPro %': round(product_counts.get('powerpro', 0) / total * 100, 1) if total > 0 else 0
            })
        
        product_df = pd.DataFrame(product_data)
        
        # Calculate summary statistics for product mentions
        total_mb = product_df['MasterBlaster'].sum()
        total_fp = product_df['FunPun'].sum()
        total_pp = product_df['PowerPro'].sum()
        total_any = product_df['Any Product'].sum()
        
        total_interactions = sum(trend_data.get('sentiment_trends', [])[i].get('total', 0) for i in range(len(time_points)))
        pct_with_products = (total_any / total_interactions * 100) if total_interactions > 0 else 0
        
        # Display product mention summary statistics
        st.markdown("#### Summary Statistics")
        
        # Use columns for a clean layout
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("MasterBlaster", total_mb)
        
        with stat_cols[1]:
            st.metric("FunPun", total_fp)
        
        with stat_cols[2]:
            st.metric("PowerPro", total_pp)
        
        with stat_cols[3]:
            st.metric("With Products", f"{pct_with_products:.1f}%")
        
        # Reshape for percentage chart (focusing on trends)
        product_pct_df = pd.melt(
            product_df,
            id_vars=['time_point', 'time_index'],
            value_vars=['MasterBlaster %', 'FunPun %', 'PowerPro %'],
            var_name='Product',
            value_name='Percentage'
        )
        
        # Create a larger, more prominent trend visualization
        product_chart = alt.Chart(product_pct_df).mark_line(
            point=True,
            strokeWidth=3  # Thicker lines for better visibility
        ).encode(
            x=alt.X('time_point:N', 
                   title='Time (External ID)', 
                   sort=None,
                   axis=alt.Axis(labelAngle=-45)),  # Angled labels for better readability
            y=alt.Y('Percentage:Q', 
                   title='Percentage', 
                   scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Product:N', 
                          legend=alt.Legend(title="Product"),
                          scale=alt.Scale(scheme='category10')),
            tooltip=['time_point:N', 'Product:N', 'Percentage:Q']
        ).properties(
            width='container',
            height=400  # Taller chart for better trend visibility
        ).interactive()
        
        # Display the chart
        st.altair_chart(product_chart, use_container_width=True)
    
    # 3. Business Impact Trends Tab
    with trend_tabs[2]:
        #st.subheader("Business Impact Trends Over Time")
        
        # Prepare data with focus on trends
        impact_data = []
        for i, point in enumerate(formatted_time_points):
            impact_counts = trend_data.get('impact_trends', [])[i]
            total = impact_counts.get('total', 1)  # Avoid division by zero
            
            # Calculate impact score for this point (weighted average)
            weights = {'high': 3, 'medium': 2, 'low': 1, 'neutral': 0}
            impact_score = sum(impact_counts.get(k, 0) * weights[k] for k in weights) / total if total > 0 else 0
            
            impact_data.append({
                'time_point': point,
                'time_index': i,  # Add index for trend line
                'High': impact_counts.get('high', 0),
                'Medium': impact_counts.get('medium', 0),
                'Low': impact_counts.get('low', 0),
                'Neutral': impact_counts.get('neutral', 0),
                'Impact Score': round(impact_score, 2),
                'High %': round(impact_counts.get('high', 0) / total * 100, 1) if total > 0 else 0,
                'Medium %': round(impact_counts.get('medium', 0) / total * 100, 1) if total > 0 else 0,
                'Low %': round(impact_counts.get('low', 0) / total * 100, 1) if total > 0 else 0,
                'Neutral %': round(impact_counts.get('neutral', 0) / total * 100, 1) if total > 0 else 0
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Calculate summary statistics for business impact
        total_high = impact_df['High'].sum()
        total_medium = impact_df['Medium'].sum()
        total_low = impact_df['Low'].sum()
        total_neutral = impact_df['Neutral'].sum()
        
        total_impacts = total_high + total_medium + total_low + total_neutral
        pct_high = (total_high / total_impacts * 100) if total_impacts > 0 else 0
        pct_medium = (total_medium / total_impacts * 100) if total_impacts > 0 else 0
        
        avg_impact_score = impact_df['Impact Score'].mean()
        
        # Display business impact summary statistics
        st.markdown("#### Summary Statistics")
        
        # Use columns for a clean layout
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("High Impact", f"{pct_high:.1f}%")
        
        with stat_cols[1]:
            st.metric("Medium Impact", f"{pct_medium:.1f}%")
        
        with stat_cols[2]:
            st.metric("Impact Score", f"{avg_impact_score:.2f}/3")
        
        with stat_cols[3]:
            most_common = max([('High', total_high), ('Medium', total_medium), 
                             ('Low', total_low), ('Neutral', total_neutral)], 
                            key=lambda x: x[1])[0]
            st.metric("Most Common", most_common)
        
        # Reshape for percentage chart (focusing on trends)
        impact_pct_df = pd.melt(
            impact_df,
            id_vars=['time_point', 'time_index'],
            value_vars=['High %', 'Medium %', 'Low %', 'Neutral %'],
            var_name='Impact Level',
            value_name='Percentage'
        )
        
        # Create a larger, more prominent trend visualization
        impact_chart = alt.Chart(impact_pct_df).mark_line(
            point=True,
            strokeWidth=3  # Thicker lines for better visibility
        ).encode(
            x=alt.X('time_point:N', 
                   title='Time (External ID)', 
                   sort=None,
                   axis=alt.Axis(labelAngle=-45)),  # Angled labels for better readability
            y=alt.Y('Percentage:Q', 
                   title='Percentage', 
                   scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Impact Level:N', 
                          legend=alt.Legend(title="Impact Level"),
                          scale=alt.Scale(
                              domain=['High %', 'Medium %', 'Low %', 'Neutral %'],
                              range=['#e74c3c', '#f39c12', '#3498db', '#95a5a6']
                          )),
            tooltip=['time_point:N', 'Impact Level:N', 'Percentage:Q']
        ).properties(
            width='container',
            height=400  # Taller chart for better trend visibility
        ).interactive()
        
        # Display the chart
        st.altair_chart(impact_chart, use_container_width=True)
    
    # 4. Sales Funnel Trends Tab
    with trend_tabs[3]:
        #st.subheader("Sales Funnel Stage Trends")
        
        # Prepare data with focus on trends
        stages = ['awareness', 'interest', 'consideration', 'intent', 'evaluation', 'purchase']
        stage_display = {
            'awareness': 'Awareness',
            'interest': 'Interest',
            'consideration': 'Consideration',
            'intent': 'Intent',
            'evaluation': 'Evaluation',
            'purchase': 'Purchase'
        }
        
        funnel_data = []
        for i, point in enumerate(formatted_time_points):
            funnel_counts = trend_data.get('funnel_stage_trends', [])[i]
            total = funnel_counts.get('total', 1)  # Avoid division by zero
            
            data_point = {'time_point': point, 'time_index': i}
            
            # Add both counts and percentages
            for stage in stages:
                data_point[stage_display[stage]] = funnel_counts.get(stage, 0)
                data_point[f"{stage_display[stage]} %"] = round(funnel_counts.get(stage, 0) / total * 100, 1) if total > 0 else 0
            
            funnel_data.append(data_point)
        
        funnel_df = pd.DataFrame(funnel_data)
        
        # Calculate summary statistics for funnel stages
        stage_counts = {}
        for stage in stages:
            stage_counts[stage_display[stage]] = funnel_df[stage_display[stage]].sum()
        
        total_funnel = sum(stage_counts.values())
        
        # Calculate key metrics
        awareness_count = stage_counts['Awareness']
        purchase_count = stage_counts['Purchase']
        
        conversion_rate = (purchase_count / awareness_count * 100) if awareness_count > 0 else 0
        
        if total_funnel > 0:
            # Find top funnel stage
            top_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        else:
            top_stage = "N/A"
        
        # Display funnel summary statistics
        st.markdown("#### Summary Statistics")
        
        # Use columns for a clean layout
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("Top Stage", top_stage)
        
        with stat_cols[1]:
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        with stat_cols[2]:
            st.metric("Purchase Count", purchase_count)
        
        with stat_cols[3]:
            st.metric("Awareness Count", awareness_count)
        
        # Reshape for percentage chart (focusing on trends)
        funnel_pct_df = pd.melt(
            funnel_df,
            id_vars=['time_point', 'time_index'],
            value_vars=[f"{stage_display[s]} %" for s in stages],
            var_name='Funnel Stage',
            value_name='Percentage'
        )
        
        # Create a larger, more prominent trend visualization
        funnel_chart = alt.Chart(funnel_pct_df).mark_line(
            point=True,
            strokeWidth=3  # Thicker lines for better visibility
        ).encode(
            x=alt.X('time_point:N', 
                   title='Time (External ID)', 
                   sort=None,
                   axis=alt.Axis(labelAngle=-45)),  # Angled labels for better readability
            y=alt.Y('Percentage:Q', 
                   title='Percentage', 
                   scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Funnel Stage:N', 
                          legend=alt.Legend(title="Funnel Stage"),
                          scale=alt.Scale(scheme='tableau10')),
            tooltip=['time_point:N', 'Funnel Stage:N', 'Percentage:Q']
        ).properties(
            width='container',
            height=400  # Taller chart for better trend visibility
        ).interactive()
        
        # Display the chart
        st.altair_chart(funnel_chart, use_container_width=True)

        
def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = {
            'all_sentences': None,
            'sentiment_dist': None,
            'impact_scores': None,
            'sentiment_impact': None,
            'funnel_metrics': None,
            'intent_dist': None,
            'high_impact': None,
            'alignment': None,
            'info_impact': None,
            'relevance_stats': None,
            'refresh_count': 0,
            'last_refresh_time': time.time(),
            'data_loaded': False,
            'relevance_filter': 'all',  # Default to showing all sentences
            'product_mentions': None
        }
    
    # Verify Redis caching
    redis_verified = verify_redis_caching()

    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    
    # Relevance filter in sidebar - changed to radio buttons for better UX
    st.session_state.data['relevance_filter'] = st.sidebar.radio(
        "Filter by Sales Funnel Relevance",
        options=["All Sentences", "Relevant Only", "Non-Relevant Only"],
        index=0,  # Default to "All Sentences"
        key="sidebar_relevance_filter"
    )
    
    # Map the selection to the actual filter value
    relevance_filter_value = "all"
    #if st.session_state.data['relevance_filter'] == "Relevant Only":
    #    relevance_filter_value = "relevant"
    #    include_non_relevant = False
    #elif st.session_state.data['relevance_filter'] == "Non-Relevant Only":
    #    relevance_filter_value = "non_relevant"
    #    include_non_relevant = True
    #else:
    #    include_non_relevant = True
    
    # Rest of the sidebar controls
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=60,
        max_value=3600,
        value=300
    )

    # Main content
    st.title("Customer Interaction Analytics Dashboard")
    #st.write("Interactive analytics with client-side filtering for fast exploration")
    
    # Check if data needs to be loaded/refreshed
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.data['last_refresh_time']
    force_refresh = st.sidebar.button("Refresh Data Now")
    
    needs_refresh = (
        st.session_state.data['all_sentences'] is None or
        force_refresh or
        (auto_refresh and time_since_refresh > refresh_interval)
    )
    
    if needs_refresh:
        # Increment refresh counter
        st.session_state.data['refresh_count'] += 1
        
        # Loading UI elements
        init_placeholder = st.empty()
        with init_placeholder.container():
            st.info("⏳ Loading or refreshing dashboard data...")
            progress_bar = st.progress(0)
        
        # Always load all sentences regardless of filter for maximum flexibility
        with st.spinner("Loading sentence data..."):
            progress_bar.progress(15)
            # Load all sentences so we can filter client-side
            st.session_state.data['all_sentences'] = fetch_all_sentences(include_non_relevant=True)
            progress_bar.progress(30)
            st.session_state.data['relevance_stats'] = fetch_relevance_stats()
            progress_bar.progress(40)
        
        # Load analytics data
        with st.spinner("Loading analytics data..."):
            progress_bar.progress(50)
            st.session_state.data['sentiment_dist'] = fetch_data("sentiment-distribution", "ml_insights:sentiment_dist")
            progress_bar.progress(60)
            st.session_state.data['impact_scores'] = fetch_data("impact-scores", "ml_insights:impact_scores")
            progress_bar.progress(65)
            st.session_state.data['sentiment_impact'] = fetch_data("sentiment-impact", "ml_insights:sentiment_impact")
            progress_bar.progress(70)
            st.session_state.data['funnel_metrics'] = fetch_data("funnel-metrics", "ml_insights:funnel_metrics")
            progress_bar.progress(75)
            st.session_state.data['intent_dist'] = fetch_data("intent-distribution", "ml_insights:intent_dist")
            progress_bar.progress(80)
            st.session_state.data['high_impact'] = fetch_data("high-impact-sentiment", "ml_insights:high_impact")
            progress_bar.progress(85)
            st.session_state.data['alignment'] = fetch_data("stage-intent-alignment", "ml_insights:alignment")
            progress_bar.progress(90)
            st.session_state.data['product_mentions'] = fetch_data("product-mentions", "ml_insights:product_mentions")
            progress_bar.progress(95)
            st.session_state.data['info_impact'] = fetch_data("impact-information", "ml_insights:info_impact")
            progress_bar.progress(100)
        
        # Update refresh timestamp and remove loading placeholder
        st.session_state.data['last_refresh_time'] = current_time
        st.session_state.data['data_loaded'] = True
        time.sleep(0.5)  # Brief pause to show completion
        init_placeholder.empty()
    
    # Display loading status
    if st.session_state.data['all_sentences']:
        # Filter the sentences based on the global relevance filter
        filtered_sentences = prepare_filtered_sentences(
            st.session_state.data['all_sentences'], 
            relevance_filter_value
        )
        
        st.success(f"✅ Data loaded successfully ({len(filtered_sentences)} sentences)")
        
    else:
        st.error("❌ Failed to load sentence data")
        st.stop()  # Stop execution if data loading failed
    
    # Create tabs for different visualizations
    tabs = st.tabs([
        "Relevance Overview",
        "Sentiment Distribution",
        "Impact Scores",
        "Sentiment Impact",
        "Funnel Metrics",
        "Intent Distribution",
        "High Impact Sentiment",
        "Business Impact Information",
        "Product Mentions",
        "Time Series Trends" 
    ])

    # Use the filtered sentences for all visualizations
    filtered_sentences = prepare_filtered_sentences(
        st.session_state.data['all_sentences'], 
        relevance_filter_value
    )

    with tabs[0]:
        # Check if relevance_stats exists in session state before using it
        if 'relevance_stats' in st.session_state.data and st.session_state.data['relevance_stats']:
            render_relevance_stats(
                st.session_state.data['relevance_stats'],
                filtered_sentences
            )
        else:
            st.warning("Sales funnel relevance statistics are not available. Please refresh the data.")
        
    with tabs[1]:
        render_sentiment_distribution(
            st.session_state.data['sentiment_dist'], 
            filtered_sentences
        )

    with tabs[2]:
        render_impact_scores(
            st.session_state.data['impact_scores'], 
            filtered_sentences
        )

    with tabs[3]:
        render_sentiment_impact(
            st.session_state.data['sentiment_impact'], 
            filtered_sentences
        )

    with tabs[4]:
        render_funnel_metrics(
            st.session_state.data['funnel_metrics'], 
            filtered_sentences
        )

    with tabs[5]:
        render_intent_distribution(
            st.session_state.data['intent_dist'], 
            filtered_sentences
        )

    with tabs[6]:
        render_high_impact_sentiment(
            st.session_state.data['high_impact'], 
            filtered_sentences
        )

    with tabs[7]:
        render_impact_information(
            st.session_state.data['info_impact'], 
            filtered_sentences
        )

    # In the main tabs section, modify the Product Mentions tab
    with tabs[8]:
        # Use the pre-loaded product mentions data from session state
        if st.session_state.data.get('product_mentions') is not None:
            render_product_mentions(
                st.session_state.data['product_mentions'], 
                filtered_sentences
            )
        else:
            st.warning("Product mentions data is not available. Please refresh the data.")
            
    # Add the new Time Series Trends tab
    with tabs[9]:
        render_trends_tab()

    # Display refresh information
    st.sidebar.write(f"Total Refreshes: {st.session_state.data['refresh_count']}")
    st.sidebar.write(f"Last Refresh: {time.strftime('%H:%M:%S', time.localtime(st.session_state.data['last_refresh_time']))}")
    
    # Auto-refresh logic
    if auto_refresh and (time.time() - st.session_state.data['last_refresh_time']) > refresh_interval:
        st.session_state.data['refresh_count'] += 1
        logger.info(f"Auto-refresh triggered. Refresh count: {st.session_state.data['refresh_count']}")
        st.rerun()
        
if __name__ == "__main__":
    main()