import sqlite3

DB_PATH = "data/project.db"
INIT_SCRIPT = "data/init.sql"
POPULATE_SCRIPT = "data/populate.sql"

def execute_sql_script(conn, script_path):
    """Execute a SQL script from a file."""
    with open(script_path, "r") as file:
        sql_script = file.read()
    conn.executescript(sql_script)
    print(f"Executed {script_path}")

def setup_database():
    """Set up the database with schema and initial data."""
    conn = sqlite3.connect(DB_PATH)
    try:
        # Run initialization script
        execute_sql_script(conn, INIT_SCRIPT)
        
        # Run population script
        execute_sql_script(conn, POPULATE_SCRIPT)

        print("Database setup complete.")
    finally:
        conn.close()

if __name__ == "__main__":
    setup_database()
