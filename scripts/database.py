import sqlite3
import pandas as pd

DB_PATH = 'data/project.db'

def get_data(query):
    """
    Fetch data from the database.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def execute_query(query):
    """
    Execute a query that doesn't return data (e.g., insert).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    conn.close()
