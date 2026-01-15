import streamlit as st
import json
import sqlite3
import html
import re
import time
import logging
from datetime import datetime
from pathlib import Path

# ===== 外部AIロジック =====
from core.anan_ai import (
    ask_question,
    load_rules_from_file,
    initialize_vector_db
)
from core.fetch_class_changes import fetch_class_changes

# ================================
# パス設定（src 基準）
# ================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# ================================
# 基本設定
# ================================
st.set_page_config(page_title="阿南高専 chatbot", page_icon="⏰")
st.markdown("""
<h1>阿南高専 学生サポートAI</h1>
<p>学校生活の疑問をすぐ解決</p>
""", unsafe_allow_html=True)

# ================================
# ログ設定（Cloud対応）
# ================================
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ================================
# CSS
# ================================
def load_css():
    try:
        with open(ASSETS_DIR / "style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        logging.warning(f"CSS load failed: {e}")

load_css()

# ================================
# SQLite（履歴）
# ================================
DB_PATH = BASE_DIR / "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_history(q, a):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (question, answer, timestamp) VALUES (?, ?, ?)",
        (q, a, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, question, answer, timestamp FROM history ORDER BY id DESC"
    )
    rows = c.fetchall()
    conn.close()
    return rows

def delete_history_item(item_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

init_db()

# ================================
# 入力検証
# ================================
MAX_LEN = 300

def validate_input(text: str):
    if not text or not text.strip():
        return False, "入力が空です"
    if len(text) > MAX_LEN:
        return False, "300文字以内で入力してください"
    if re.search(r"[<>]", text):
        return False, "使用できない文字が含まれています"
    return True, ""

# ================================
# DoS対策（連打防止）
# ================================
def rate_limit(sec=5):
    now = time.time()
    last = st.session_state.get("last_request_time", 0)
    if now - last < sec:
        st.warning("少し待ってから再度実行してください")
        return False
    st.session_state.last_request_time = now
    return True

# ================================
# データ読み込み（RAG）
# ================================
@st.cache_resource
def load_all_data():
    with open(DATA_DIR / "timetable1.json", encoding="utf-8") as f:
        timetable = json.load(f)

    def load_db(path: Path):
        return initialize_vector_db(load_rules_from_file(path))

    return {
        "timetable": timetable,
        "grooming": load_db(DATA_DIR / "style.txt"),
        "grades": load_db(DATA_DIR / "grade.txt"),
        "abstract": load_db(DATA_DIR / "abstract.txt"),
        "cycle": load_db(DATA_DIR / "cycle.txt"),
        "abroad": load_db(DATA_DIR / "abroad.txt"),
        "sinro": load_db(DATA_DIR / "sinro.txt"),
        "part": load_db(DATA_DIR / "part.txt"),
        "other": load_db(DATA_DIR / "other.txt"),
        "money": load_db(DATA_DIR / "money.txt"),
        "domitory": load_db(DATA_DIR / "domitory.txt"),
        "clab": load_db(DATA_DIR / "clab.txt"),
    }

dbs = load_all_data()

# intent → VectorDB の対応表
DB_MAP = {
    "grooming": dbs["grooming"],
    "grades": dbs["grades"],
    "abstract": dbs["abstract"],
    "cycle": dbs["cycle"],
    "abroad": dbs["abroad"],
    "sinro": dbs["sinro"],
    "part": dbs["part"],
    "money": dbs["money"],
    "domitory": dbs["domitory"],
    "clab": dbs["clab"],
}

# ================================
# 管理者認証
# ================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.markdown("### 管理者")
    pin = st.text_input("管理者PIN", type="password")
    admin_pin = st.secrets.get("ADMIN_PIN")
    if admin_pin and pin == admin_pin:
        st.session_state.is_admin = True
        st.success("管理者モード")

# ================================
# ページ管理
# ================================
if "page" not in st.session_state:
    st.session_state.page = "home"

def nav_button(label, target):
    if st.button(label, key=f"nav_{target}"):
        st.session_state.page = target
        st.rerun()

st.markdown("## 🔽 機能を選択してください")
c1, c2, c3, c4 = st.columns(4)
with c1: nav_button("🏠 ホーム", "home")
with c2: nav_button("💬 チャット", "chat")
with c3: nav_button("🔄 授業変更", "change")
with c4: nav_button("📜 履歴", "history")

page = st.session_state.page

# ================================
# ホーム
# ================================
if page == "home":
    st.info("学内向け試験運用版です")

# ================================
# チャット
# ================================
elif page == "chat":
    q = st.text_input("質問を入力してください")

    if st.button("送信"):
        if not rate_limit():
            st.stop()

        ok, msg = validate_input(q)
        if not ok:
            st.error(msg)
            st.stop()

        with st.spinner("考えています..."):
            ans = ask_question(q, DB_MAP)

        st.success(ans)
        add_history(html.escape(q), html.escape(ans))

# ================================
# 授業変更
# ================================
elif page == "change":
    c = st.text_input("クラス（例：3I）")
    if st.button("取得"):
        result = fetch_class_changes(c if c else None)
        st.info(result)
        add_history(c or "全体", html.escape(result))

# ================================
# 履歴
# ================================
elif page == "history":
    if st.session_state.is_admin:
        if st.button("🗑️ 履歴をすべて削除"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            st.rerun()

    history = load_history()
    if not history:
        st.info("履歴はありません")

    for h_id, q, a, t in history:
        t_jp = datetime.fromisoformat(t).strftime("%Y/%m/%d %H:%M")
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.caption(t_jp)

        if st.session_state.is_admin:
            if st.button("削除", key=f"del_{h_id}"):
                delete_history_item(h_id)
                st.rerun()
