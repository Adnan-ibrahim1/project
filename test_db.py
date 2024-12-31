from scripts.database import fetch_data

# Query the database
query = "SELECT * FROM dataset"
data = fetch_data(query)

# Print the data
print("Dataset loaded from database:")
print(data)
