import sqlite3
import config


def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS file_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT,
            embedding BLOB,
            modality TEXT DEFAULT 'text',
            updated_at TIMESTAMP,
            UNIQUE(file_path, chunk_index)
        )
    ''')
    # Migrate older databases that lack the modality column
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(file_index)")]
    if "modality" not in cols:
        conn.execute("ALTER TABLE file_index ADD COLUMN modality TEXT DEFAULT 'text'")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized with WAL mode.")
