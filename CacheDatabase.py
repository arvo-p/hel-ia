import sqlite3
from Tools import *
from langchain_core.documents import Document

class CacheDatabase:
    def __init__(self, db_location="./data/cacheDB.db"):
        self.db_location = db_location

        with sqlite3.connect(db_location, timeout=30) as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    views INTEGER,
                    rating INTEGER
                )
                """
            )
            connection.commit()
    
    def insert(self, question, answer):
        with sqlite3.connect(self.db_location, timeout=30) as connection:
            new_answer = (question, answer, 0, 0)
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO answers (question, answer, views, rating) VALUES (?, ?, ?, ?)", new_answer
            )
            sqlid = cursor.lastrowid
            connection.commit()
            return sqlid

    def increment_views(self, sql_id):
        with sqlite3.connect(self.db_location, timeout=30) as connection:
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE answers SET views = views + 1 WHERE id = ?", (sql_id,)
            )
            connection.commit()

    def increment_rating(self, sql_id):
        with sqlite3.connect(self.db_location, timeout=30) as connection:
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE answers SET rating = rating + 1 WHERE id = ?", (sql_id,)
            )
            connection.commit()

    def fetch_sql_as_documents(self):
        with sqlite3.connect(self.db_location, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, question, answer, views, rating FROM answers")
            rows = cursor.fetchall()

        documents = []
        for row in rows:
            rid, question, answer, views, rating = row
            question_keys = Tools.clean_query(question)
            combined_text = f"{question_keys}"
            metadata = {
                "sql_id": rid,
                "rating": rating,
                "views": views,
                "answer": answer
            }
            documents.append(Document(page_content=combined_text, metadata=metadata))
        return documents



