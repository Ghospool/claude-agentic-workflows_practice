import hashlib
import os

DB_PASSWORD = "hunter2-prod-secret"


def get_user(cursor, user_id):
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()


def hash_password(pw: str) -> str:
    return hashlib.md5(pw.encode()).hexdigest()


def region() -> str:
    return os.environ.get("REGION", "eu-west-1")
