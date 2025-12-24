# AI-based-Drug-Identifier-and-Smart-Drug-Assistant-Backend

Feng 497 Graduation Project - An intelligent drug identifier and smart assistant powered by RAG (Retrieval Augmented Generation) and LLM.

## 🚀 Quick Start with Docker

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- At least 8GB RAM available
- 10GB free disk space (for models)

### Starting the Application

```bash
# Clone the repository
git clone <repo-url>
cd AI-based-Drug-Identifier-and-Smart-Drug-Assistant-Backend

# Build and start all services
docker-compose up --build
```

This command will automatically:
1. **Build** the FastAPI application Docker image
2. **Start PostgreSQL** database (port 5432)
3. **Start Ollama** LLM service (port 11434) and download required models
4. **Start FastAPI** backend API (port 8000)

### 📋 Available Services

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI API | 8000 | Backend API with health checks |
| PostgreSQL | 5432 | User & drug interaction database |
| Ollama | 11434 | LLM inference engine |

### 🌐 Access Points

Once running, access the application at:

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/check-db

### 💊 Populate Pill Database (Required for Pill Matching)

The pill identification endpoint can match predictions to the `pills` table. After starting Docker, seed the dataset once:

```bash
# Run the migration inside the API container (recommended)
docker compose exec -T api python -m smart_drug_assistant.db.migration
```

This loads `smart_drug_assistant/db/pill_color_imprint_dataset.csv` into PostgreSQL.

### 📄 Adding Medical Leaflets

Place patient information leaflets (PDFs) for the RAG system:

```bash
mkdir -p patient_leaflets
# Copy your PDF files to the directory
cp /path/to/medical/leaflets/*.pdf patient_leaflets/
```

Medical documents will be automatically processed and indexed on first API usage.

### 🧪 Test the API

```bash
# Test authentication endpoint
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123"
  }'

# Query the RAG system for drug information
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the side effects of this medication?"
  }'

# Pill identification (image upload)
curl -X POST "http://localhost:8000/pill_identify/" \
  -F "file=@/path/to/pill.jpg"
```

### 🔎 Pill Identification Matching Output

`POST /pill_identify/` returns:

- **Predictions**: shape, colors, and an OCR imprint
- **DB info**: chosen color filters and current `pills` row count
- **Matches**: ranked candidate pills from the DB

The matcher is designed to be tolerant to imperfect OCR/color predictions by using canonical imprint formatting and fuzzy matching.

### 🛑 Stopping Services

```bash
# Stop all services (keep data)
docker-compose down

# Stop and remove all data/volumes
docker-compose down -v
```

### 📊 View Logs

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api         # Backend API
docker-compose logs -f ollama      # LLM service
docker-compose logs -f postgres    # Database
```

## 🔧 Advanced Configuration

### Development Mode
For development with automatic reload:
```bash
docker-compose up
```
Code changes will automatically reload the API.

### Production Deployment
For production deployment, use the production compose file:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

This enables:
- Restart policies
- Resource limits (CPU/memory)
- Persistent volumes
- Optimized configurations

### Troubleshooting

**Port Already in Use**
```bash
# Kill process using port (e.g., 8000)
lsof -ti:8000 | xargs kill -9

# Or change ports in docker-compose.yml
```

**Models Not Downloading**
```bash
# Manually pull Ollama models
docker exec smart_drug_ollama ollama pull neural-chat
docker exec smart_drug_ollama ollama pull nomic-embed-text
```

**Reset Database**
```bash
# Remove data and reinitialize
docker-compose down -v
docker-compose up --build
```

## 📖 More Information

For detailed Docker setup and troubleshooting, see [DOCKER.md](DOCKER.md).

For details on the pill matching logic and why it works this way, see [PILL_MATCHING_LOGIC.md](PILL_MATCHING_LOGIC.md).

## 🏗️ Project Structure

```
├── smart_drug_assistant/
│   ├── api/              # FastAPI application
│   ├── core/             # Configuration & security
│   ├── db/               # Database setup & migrations
│   ├── models/           # SQLAlchemy models
│   ├── routers/          # API endpoints
│   └── schemas/          # Pydantic schemas
├── patient_leaflets/     # Medical documents for RAG
├── chroma_db/            # Vector database
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Development services
└── requirements.txt      # Python dependencies
```

## 🤖 Technology Stack

- **Backend**: FastAPI + Uvicorn
- **Database**: PostgreSQL
- **LLM**: Ollama (neural-chat)
- **Embeddings**: Nomic Embed Text
- **Vector Store**: Chroma DB
- **Authentication**: JWT + bcrypt
- **ORM**: SQLAlchemy

## 📝 License

Feng 497 Graduation Project
