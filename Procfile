# Create a Procfile in your project root
# Procfile
#redis_service: cd C:\Users\rapha\Downloads\Redis-x64-3.0.504 && redis-server.exe
db_service: cd ml_insights_system/db_service && poetry run python -m src.db_service.service
queue_service: cd ml_insights_system//queue_service && poetry run python src/queue_service/service.py
worker_service: cd ml_insights_system//worker_service && poetry run python src/worker_service/worker.py
api_service: cd ml_insights_system//api_gateway && poetry run uvicorn src.api_gateway.app:app --host 0.0.0.0 --port 8000 --reload
dashboard_service: cd ml_insights_system//dashboard_service && poetry run streamlit run src/dashboard_service/app.py
feature_service: cd ml_insights_system//feature_service && poetry run uvicorn src.feature_service.service:app --host 0.0.0.0 --port 8003
