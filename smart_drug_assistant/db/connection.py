import os

from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as PGConnection
from typing import Optional

load_dotenv()

def get_db_connection(
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[str] = None
) -> PGConnection:
    dbname = dbname or os.getenv("PG_DBNAME")
    user = user or os.getenv("PG_USER")
    password = password or os.getenv("PG_PASSWORD")
    host = host or os.getenv("PG_HOST")
    port = port or os.getenv("PG_PORT")
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    return conn
