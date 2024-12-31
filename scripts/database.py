import sqlite3
import pandas as pd

DB_PATH = "data/project.db"

def connect_to_db():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def fetch_data(query: str) -> pd.DataFrame:
    """Fetch data from the database and return as a pandas DataFrame."""
    conn = connect_to_db()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()
