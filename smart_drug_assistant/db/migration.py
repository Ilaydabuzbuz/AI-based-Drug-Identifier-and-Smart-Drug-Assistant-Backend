import psycopg
import pandas as pd

# Connect to DB
conn = psycopg.connect(
    dbname="smart_drug_db",
    user="drug_user",
    password="drug_password",
    host="localhost",
    port="5432"
) 

# Create table if not exists
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pills (
            id SERIAL PRIMARY KEY,
            image_path TEXT,
            image_filename TEXT,
            pill_name TEXT,
            shape TEXT,
            color TEXT,
            imprint TEXT
        );
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            hashed_password TEXT NOT NULL
        );
    """)
    conn.commit()

# Load CSV
df = pd.read_csv("smart_drug_assistant/db/pill_color_imprint_dataset.csv")

# Insert data
for _, row in df.iterrows():
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pills (image_path, image_filename, pill_name, shape, color, imprint)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row.image_path, row.image_filename, row.pill_name, row.shape, row.color, row.imprint))
conn.commit()
conn.close()
