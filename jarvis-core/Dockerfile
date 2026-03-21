FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx pydantic python-dotenv \
    qdrant-client sentence-transformers ddgs redis \
    google-auth google-api-python-client \
    apscheduler pytz yfinance python-multipart
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
