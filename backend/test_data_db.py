
from database import get_db_connection

connection = None

try:
    connection = get_db_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT DATABASE(), CURRENT_USER()")

    result = cursor.fetchone()
    print("Database connection successful!")
    print("Database and user:", result)

    cursor.close()

except Exception as error:
    print("Database connection failed.")
    print(type(error).__name__, error)

finally:
    if connection is not None and connection.is_connected():
        connection.close()
