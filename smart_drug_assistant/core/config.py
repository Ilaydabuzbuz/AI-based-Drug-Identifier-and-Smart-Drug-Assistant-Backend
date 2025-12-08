import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Drug Identifier API"
    PROJECT_VERSION: str = "1.0.0"
    
    # Database
    PG_USER: str = os.getenv("PG_USER", "postgres")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD", "password")
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: str = os.getenv("PG_PORT", "5432")
    PG_DBNAME: str = os.getenv("PG_DBNAME", "pill_data")
    
    DATABASE_URL: str = f"postgresql+pg8000://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()
