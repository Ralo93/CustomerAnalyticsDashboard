# voiceLine Task 3

## System Architecture
## Overview
This document describes the architecture of my data processing and visualization system. The system consists of multiple components that work together to process, store, and visualize data from the sales environment.

## Classification Targets

```mermaid
mindmap
  root((AI Classification))
    Sales Funnel Relevance
      Awareness
      Interest
      Retention
        Consideration
        Evaluation
      Intent
      Purchase
      
    Communication Intent
      Complaint
      Feedback
      General
      Information
      Purchase
      Support
    
    Sentiment
      Positive
      Neutral
      Negative
    
    Business Impact
      High
      Low
      Critical
      Medium
      Neutral
```

classDef default fill:#f9f9f9,stroke:#999,stroke-width:1px,color:#333
classDef root fill:#6f42c1,color:white,stroke:#5a32a3


## Sneak 

### Sales Funnel Classification
![newplot](https://github.com/user-attachments/assets/3377d4e9-cf05-48cc-a5f8-32a7dc110f01)

### Sentiment per Sales Funnel
![newplot (4)](https://github.com/user-attachments/assets/aefb6cc1-0f86-43e0-a536-a67b10db1912)

### Product Mentions
![newplot (3)](https://github.com/user-attachments/assets/6f54f67f-9334-4d24-be74-06393dd15c70)

### Word Count per Sentence
![newplot (2)](https://github.com/user-attachments/assets/c9ccc8bf-74d1-46ad-8a1e-9a630171573e)

### Business Impact 
![newplot (1)](https://github.com/user-attachments/assets/b41825f1-48d9-4dd7-9f23-0e283b2e5e2a)




## Architecture Diagram

```mermaid
flowchart TD
    User([User]) -->|Request| API[API Gateway]
    
    API -->|Forward Request| FE[Feature Extractor]
    API -->|Enqueue Task| MQ[(Message Queue)]
    API -->|Store Raw Data| DBS[DB Service]
    
    FE -->|Send Extracted Features| DBS
    
    MQ -->|Dequeue Task| Worker[Worker]
    
    Worker -->|OpenAI API Call| OpenAI[OpenAI Service]
    OpenAI -->|Response| Worker
    
    Worker -->|Send Results| DBS
    
    DBS -->|Store All Data| DB[(Database)]
    
    DB -->|Cache Data| Redis[(Redis Cache)]
    
    Redis -->|Serve Data| Dashboard[Dashboard]
    
    %% GitHub-friendly styling
    classDef default fill:#f9f9f9,stroke:#999,stroke-width:1px,color:#333
    classDef service fill:#0366d6,stroke:#0366d6,color:white
    classDef storage fill:#28a745,stroke:#28a745,color:white
    classDef external fill:#6f42c1,stroke:#6f42c1,color:white
    classDef user fill:#d73a49,stroke:#d73a49,color:white
    
    class API,FE,Worker,DBS service
    class DB,Redis,MQ storage
    class OpenAI external
    class User,Dashboard user
```



### API Gateway

- Serves as the entry point for all user requests
- Forwards requests to the Feature Extractor for processing
- Sends raw data to the DB Service for storage
- Enqueues tasks in the Message Queue for asynchronous processing

### Feature Extractor

- Processes input data to extract relevant features
- Stores extracted features in the database
- Optimized for rapid feature analysis and extraction

### DB Service

- Receives raw input data from the API Gateway
- Responsible for storing unprocessed data in the database
- Handles database connections and transactions

### Message Queue

- Maintains a queue of tasks to be processed asynchronously
- Provides reliable task delivery to the Worker
- Supports prioritization and retry mechanisms

### Worker

- Consumes tasks from the Message Queue
- Makes API calls to OpenAI for advanced processing
- Stores processing results back to the database

### Database

- Central data store for the entire system
- Stores raw input data, extracted features, and processing results
- Provides persistent storage with data integrity guarantees

### Redis Cache

- Caches frequently accessed data from the database
- Reduces database load and improves dashboard performance
- Implements efficient invalidation strategies (not yet)

### Dashboard

- Provides visualization of processed data
- Retrieves data from Redis cache for optimal performance
- Offers interactive data exploration capabilities

### Data Flow

- User sends a request to the API Gateway
  
1. The API Gateway forwards the request to the Feature Extractor
2. Sends raw data to the DB Service
3. Enqueues a task in the Message Queue
4. Feature Extractor processes the data and stores features in the database
5. DB Service stores the raw data in the database
6. Worker pulls tasks from the Message Queue
7. Worker makes API calls to OpenAI and stores results in the database
8. Redis caches relevant data from the database
9. Dashboard retrieves data from Redis to display visualizations


## TechStack

- Python 3.12
- Nginx
- Poetry
- Streamlit
- PostgreSQL
- RabbitMQ
- RESTfulAPI
- Docker

## Deployment Considerations

- Each component can be deployed as a separate microservice
- Database and Redis should be configured for high availability
- Consider using auto-scaling for API Gateway, Worker, and Feature Extractor based on load
- Implement appropriate monitoring and alerting for all components

## Future Enhancements

1. Add service specific databases and db_services
2. Implement more sophisticated caching strategies
3. Add real-time processing capabilities
4. Expand dashboard functionality with additional visualization options

# Setup

1. Start a redis server installed and started locally
2. Have postgre server installed locally
3. Have rabbitmq installed and running locally
4. Clone repository

# Application Startup
1. Navigate to root of project
2. have honcho installed with pip install honcho
3. Run honcho start


# Added lines (dont take seriously)

```plaintext
111	I hate everything. My life sucks and I will destroy this company with a bomb.
112	FOR YOR INFORMATION: This is critical business impact, very critical! But I dont tell you why, still very importante!
113	I will definetly buy a 1000000 of FunPun and 500 of PowerPro.
110	HELLO! HELP ME! I NEED URGENT HELP! THIS IS NOT A DRILL, SEND HELP!.
```
