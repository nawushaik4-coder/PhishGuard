
import json
import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "phishguard_db"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connection_timeout=5,
    )


def save_scan(url: str, result: dict):
    """Save one scan result to MySQL."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO scan_history
                (url, verdict, risk_score, reasons)
            VALUES (%s, %s, %s, %s)
        """

        values = (
            url,
            result["verdict"],
            result["risk_score"],
            json.dumps(result["reasons"]),
        )

        cursor.execute(query, values)
        connection.commit()

        return cursor.lastrowid

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_scan_history(limit: int = 20):
    """Return the most recent scan records."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT scan_id, url, verdict, risk_score, reasons, created_at
            FROM scan_history
            ORDER BY scan_id DESC
            LIMIT %s
            """,
            (limit,),
        )

        records = cursor.fetchall()

        for record in records:
            if isinstance(record["reasons"], str):
                record["reasons"] = json.loads(record["reasons"])

        return records

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
