from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from smart_drug_assistant.db.connection import get_db_connection
# from smart_drug_assistant.routers import rag  # Temporarily disabled - requires langchain
from smart_drug_assistant.routers import auth, users

'''
SELAM
docker yükleyin
terminalde : Run docker-compose up to start PostgreSQL
             Run uvicorn smart_drug_assistant.api.main:app --reload
             to start the FastAPI server.
'''
app = FastAPI(title="Medical Leaflet RAG API")

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(rag.router)  # Temporarily disabled - requires langchain
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(users.router, prefix="/users", tags=["users"])

@app.get("/check-db")
def check_db():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "success", "message": "Database connection successful."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
