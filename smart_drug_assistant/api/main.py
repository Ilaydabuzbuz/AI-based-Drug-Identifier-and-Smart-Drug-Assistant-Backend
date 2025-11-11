from fastapi import FastAPI
from smart_drug_assistant.db.connection import get_db_connection

'''
SELAM
docker yükleyin
terminalde : Run docker-compose up to start PostgreSQL
             Run uvicorn smart_drug_assistant.api.main:app --reload
             to start the FastAPI server.
'''
app = FastAPI()

@app.get("/check-db")
def check_db():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "success", "message": "Database connection successful."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
