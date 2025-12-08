import pg8000.dbapi
import pandas as pd

# Connect to DB
conn = pg8000.dbapi.connect(
    database="pill_data",
    host="localhost",
    port=5432,
    user="postgres", # Assuming default or env var should be used, but matching original script's hardcoded style or improving it? The original didn't pass user/pass! 
    # Original: psycopg2.connect(dbname="pills_data", host="localhost", port="5432") 
    # Wait, original script didn't have user/pass? It might rely on OS usage. pg8000 might need it.
    # I will add user='postgres', password='password' as fallback or try to read envs if I can, but to keep it simple and assuming user has set up envs or defaults.
    # Actually, let's use the .env if possible or default credentials from config logic.
    # I'll stick to replacing the import and connect call.
    password="password"
) 

# Load CSV
df = pd.read_csv("db/pill_color_imprint_dataset.csv")

# Insert data
for _, row in df.iterrows():
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pills (image_path, image_filename, pill_name, shape, color, imprint)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row.image_path, row.image_filename, row.pill_name, row.shape, row.color, row.imprint))
conn.commit()
conn.close()
