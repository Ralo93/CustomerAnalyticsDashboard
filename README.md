# voiceLine Task 3

## Overview
This document describes the architecture of my data processing and visualization system. The system consists of multiple components that work together to process, store, and visualize data from the sales environment.

## Classification Targets

To make sense of the provided sentences, I extracted features and provided labels in the form of classifications. Lets start with the classifications.
I simply used OpenAIs 3.5-turbo model as a baseline. Classification of each sentence was done using four major dimensions:

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


## Labels Sneak Preview


![sfs](https://github.com/user-attachments/assets/aa0e7250-79e2-40fd-bafa-f8ece7dd3ace)


![sentimentperstage](https://github.com/user-attachments/assets/da4b55c4-ac43-4f62-be2a-2fc87a5373ab)


![bi](https://github.com/user-attachments/assets/81c8c2e8-3ef4-40fc-8951-471ad7d0d62e)


## Features Sneak Preview

![pm](https://github.com/user-attachments/assets/3ad617e6-000e-4a5c-b4cd-fd030c0423fa)


![wordcount](https://github.com/user-attachments/assets/9c2e91aa-88a9-4acc-80bf-220aafda3367)




# As a CTO, I also focus heavily on non-functional requirements and product strategy.

1. Maintainability
2. Scalability
3. Robustness
4. Test Coverage
5. Deployment ready architecture
6. Fast Iteration using Customer Feedback asap

I need an architecture which is fast, can handle changing requirements, can be scaled and is cheap. I want to iterate quickly to pivot into better product strategies.
So I came up with the following architecture for an MVP:

- Input: Sentences
- Output: Interactive Dashboard 

## Architecture Diagram

```mermaid
flowchart TD
    User([User]) -->|Request| API[API Gateway]
    
    API -->|Forward Request| FE[Feature Extractor]
    API -->|Enqueue Task| MQ[(Message Queue)]
    API -->|Store Raw Data| DBS[DB Service]
    
    FE -->|Send Extracted Features| DBS
    
    MQ -->|Dequeue Task| Worker[Worker]
    
    Worker -->|OpenAI API Call| OpenAI[OpenAI Service or Internal Model] 
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

## Components

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
  
1. The API Gateway
   - forwards the request to the Feature Extractor
   - Sends raw data to the DB Service
   - Enqueues a task in the Message Queue
3. Feature Extractor processes the data synchronously and stores features in the database
4. DB Service stores the raw data in the database
5. Worker pulls tasks from the Message Queue
6. Worker makes API calls to OpenAI and stores results in the database asynchronously
7. Redis caches relevant data from the database
8. Dashboard retrieves data from Redis to display visualizations


## TechStack

- Python 3.12
- Nginx
- Poetry
- Streamlit
- PostgreSQL
- RabbitMQ
- RESTfulAPI
- Docker


## Dashboard Sneak Preview

![relevance](https://github.com/user-attachments/assets/ea6a8e11-ac32-467e-924b-af9b1b120bcd)

![funnel](https://github.com/user-attachments/assets/3af951fe-4d9d-4dbe-b5ea-04ef8eb71547)

![ProductMentions](https://github.com/user-attachments/assets/482884b8-06e5-4d55-a5d3-55ac1e29f457)

![Screenshot 2025-02-26 113755](https://github.com/user-attachments/assets/10b87f9c-4e23-43ea-8372-d0eaed6e03a7)


## Deployment Considerations

- Each component can be deployed as a separate microservice
- Database and Redis should be configured for high availability
- Consider using auto-scaling for API Gateway, Worker, and Feature Extractor based on load
- Implement appropriate monitoring and alerting for all components

## Future Enhancements

0. Over time create customer - client specific sales funnels and models
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


```
