import sqlite3
from datetime import datetime
from pathlib import Path

# ===============================
# DBパス（src/history.db）
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "history.db"

# ===============================
# DB接続
# ===============================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# ===============================
# DB初期化（起動時1回）
# ===============================
def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL,
            page TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# ===============================
# 履歴保存
# ===============================
def save_history(page: str, question: str, answer: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO history (time, page, question, answer)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            page,
            question,
            answer,
        ),
    )
    conn.commit()
    conn.close()

# ===============================
# 履歴取得
# ===============================
def load_history(limit: int = 50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, time, page, question, answer
        FROM history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

# ===============================
# 履歴全削除（管理者）
# ===============================
def clear_history():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()

# ===============================
# 履歴1件削除（管理者）
# ===============================
def delete_history_by_id(history_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE id = ?", (history_id,))
    conn.commit()
    conn.close()
