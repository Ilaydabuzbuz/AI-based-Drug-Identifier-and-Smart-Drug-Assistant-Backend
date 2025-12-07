from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from psycopg import Connection
from smart_drug_assistant.db.connection import get_db_connection
from smart_drug_assistant.models.user import UserCreate, UserResponse, Token
from smart_drug_assistant.core.security import get_password_hash, verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, conn: Connection = Depends(get_db)):
    # Check if existing
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE username = %s", (user.username,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username already registered")
        
        hashed_pw = get_password_hash(user.password)
        cur.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s) RETURNING id, username, email",
            (user.username, user.email, hashed_pw)
        )
        new_user = cur.fetchone()
        conn.commit()
        
        return UserResponse(id=new_user[0], username=new_user[1], email=new_user[2])

@router.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), conn: Connection = Depends(get_db)):
    with conn.cursor() as cur:
        cur.execute("SELECT username, hashed_password FROM users WHERE username = %s", (form_data.username,))
        user = cur.fetchone()
        
    if not user or not verify_password(form_data.password, user[1]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user[0]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
