import psycopg2
import pandas as pd

# Connect to DB
conn = psycopg2.connect(
    dbname="pill_data",
    host="localhost",
    port="5432"
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
