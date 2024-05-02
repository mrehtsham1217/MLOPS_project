from dotenv import load_dotenv
import pymysql
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

# Connect to MySQL database
connection = pymysql.connect(
    host=host,
    user=user,
    password=password,
    db=db
)

# Create cursor
cursor = connection.cursor()

# Execute SQL query
query = 'SELECT * FROM raw'
cursor.execute(query)

# Fetch and print rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close cursor and connection
cursor.close()
connection.close()
