import psycopg2
import pandas as pd

def connect_to_db():
    return psycopg2.connect(
        dbname="adtb",
        user="adnan",
        password="04152005",
        host="localhost",
        port="5432"
    )

def fetch_data(query):
    conn = connect_to_db()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
