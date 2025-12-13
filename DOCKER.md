# Docker Setup Guide

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Clone the repository
git clone <repo-url>
cd AI-based-Drug-Identifier-and-Smart-Drug-Assistant-Backend

# Build and start all services
docker-compose up --build
```

This will start:
- **PostgreSQL** (port 5432)
- **Ollama** (port 11434) - with models auto-pulled
- **FastAPI** (port 8000)

### 2. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Database Check**: http://localhost:8000/check-db

### 3. Add Medical Leaflets

Place your PDF files in the `patient_leaflets/` folder:

```bash
mkdir -p patient_leaflets
# Copy your PDFs here
cp /path/to/leaflets/*.pdf patient_leaflets/
```

The system will automatically process and embed them on first use.

### 4. Test the RAG API

```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the side effects of this medication?"}'
```

## Service Details

### PostgreSQL
- **Host**: postgres (internal) / localhost:5432 (external)
- **Database**: smart_drug_db
- **User**: drug_user
- **Password**: drug_password

### Ollama
- **Host**: ollama (internal) / localhost:11434 (external)
- **Pre-installed Models**:
  - `neural-chat` - LLM for responses
  - `nomic-embed-text` - Embeddings

### FastAPI
- **Host**: api (internal) / localhost:8000 (external)
- **Reload**: Enabled for development (remove `--reload` in production)

## Stopping Services

```bash
# Stop all services
docker-compose down

# Remove all data (volumes)
docker-compose down -v
```

## Troubleshooting

### Port Already in Use
```bash
# Change ports in docker-compose.yml if needed
# Then rebuild
docker-compose up --build
```

### Models Not Downloading
```bash
# Manually pull models
docker exec smart_drug_ollama ollama pull neural-chat
docker exec smart_drug_ollama ollama pull nomic-embed-text
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ollama
docker-compose logs -f postgres
```

### Database Issues
```bash
# Reset database
docker-compose down -v
docker-compose up
```

## Production Considerations

For production deployment:

1. **Update `.env`** with secure credentials
2. **Change Ollama GPU settings** if you have NVIDIA GPU available (set `OLLAMA_NUM_GPU=1`)
3. **Use production ASGI server** (gunicorn with uvicorn workers)
4. **Enable SSL/TLS** with nginx reverse proxy
5. **Use persistent volumes** for database and Ollama data
6. **Set resource limits** in docker-compose.yml

Example production-ready command:
```bash
CMD ["gunicorn", "smart_drug_assistant.api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```
