from sqlalchemy import text
from app.data.db_engine import init_db, engine


def test_users_table_has_password_hash_column():
    init_db()
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(users)")).mappings().all()
    columns = {str(r["name"]) for r in rows}
    assert "password_hash" in columns
