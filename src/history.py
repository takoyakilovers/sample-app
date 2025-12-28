import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")

def get_conn():
    return sqlite3.connect(DB_PATH)

# ===============================
# DB初期化（必ず呼ばれる）
# ===============================
def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            page TEXT,
            question TEXT,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()

# ===============================
# 履歴保存
# ===============================
def save_history(page, question, answer):
    init_db()  # ★ 念のため毎回
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (time, page, question, answer) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), page, question, answer)
    )
    conn.commit()
    conn.close()

# ===============================
# 履歴取得
# ===============================
def load_history(limit=50):
    init_db()  # ★ ここが重要
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT time, page, question, answer FROM history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

# ===============================
# 履歴削除
# ===============================
def clear_history():
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()
