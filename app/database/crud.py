from database.db import conn
import pandas as pd

def save_record(name, age, gender, prediction, confidence):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients
        (name, age, gender, prediction, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (name, age, gender, prediction, confidence))
    conn.commit()


def get_records():
    return pd.read_sql_query(
        "SELECT * FROM patients ORDER BY id DESC",
        conn
    )


def clear_records():
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patients")
    conn.commit()
import pandas as pd
from database.db import conn

def get_statistics():
    history = pd.read_sql_query(
        "SELECT * FROM patients",
        conn
    )

    if history.empty:
        return {
            "total": 0,
            "avg_confidence": 0,
            "most_common": "N/A"
        }

    total = len(history)

    avg_confidence = history["confidence"].mean() * 100

    most_common = history["prediction"].mode()[0]

    return {
        "total": total,
        "avg_confidence": avg_confidence,
        "most_common": most_common
    }