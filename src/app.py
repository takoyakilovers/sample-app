import streamlit as st
import json
import sqlite3
import html
import re
import time
import logging
from datetime import datetime

# ===== 外部AIロジック =====
from anan_ai import (
    ask_question,
    load_rules_from_file,
    initialize_vector_db
)

# 授業変更
from fetch_class_changes import fetch_class_changes

# ================================
# 基本設定
# ================================
st.set_page_config(page_title="阿南高専 chatbot",page_icon="⏰")
st.markdown("""
<h1>阿南高専 学生サポートAI</h1>
<p>学校生活の疑問をすぐ解決</p>
""", unsafe_allow_html=True)

# ================================
# ログ設定
# ================================
logging.basicConfig(
    filename="app.log",
    level = logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ================================
# CSS
# ================================
def load_css():
    try:
        with open("style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        logging.warning(f"CSS load failed: {e}")

load_css()


# ================================
# SQLite
# ================================
def init_db():
    conn = sqlite3.connect("history.db")
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
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (question, answer, timestamp) VALUES (?, ?, ?)",
        (q, a, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    # 削除用に id も取得するように変更
    c.execute("SELECT id, question, answer, timestamp FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# ★ 追加: 指定IDの履歴を削除する関数
def delete_history_item(item_id):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

init_db()

# ================================
# 入力検証
# ================================
MAX_LEN = 300

def validate_input(text):
    if not text or not text.strip():
        return False, "入力が空です"
    if len(text) > MAX_LEN:
        return False, "300文字以内で入力してください"
    if re.search(r"[<>]",text):
        return False, "使用できない文字が含まれています"
    return True, ""

# ================================
# DoS対策 (連打防止)
# ================================
def rate_limit(sec=5):
    now = time.time()
    last = st.session_state.get("last_request_time",0)
    if now - last < sec:
        st.warning("少し待ってから再度実行してください")
        return False
    st.session_state.last_request_time = now
    return True

# ================================
# データ読み込み
# ================================
@st.cache_resource
def load_all_data():
    # ※ ファイルパスなどは環境に合わせてください
    with open("data/timetable1.json", "r", encoding="utf-8") as f:
        timetable = json.load(f)

    def load_db(path):
        return initialize_vector_db(load_rules_from_file(path))

    return {
        "timetable": timetable,
        "grooming": load_db("data/style.txt"),
        "grades": load_db("data/grade.txt"),
        "abstract": load_db("data/abstract.txt"),
        "cycle": load_db("data/cycle.txt"),
        "abroad": load_db("data/abroad.txt"),
        "sinro": load_db("data/sinro.txt"),
        "part": load_db("data/part.txt"),
        "other": load_db("data/other.txt"),
        "money": load_db("data/money.txt"),
        "domitory": load_db("data/domitory.txt"),
        "clab": load_db("data/clab.txt"),
    }

dbs = load_all_data()

# ================================
# 管理者認証
# ================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.markdown("### 管理者")
    pin = st.text_input("管理者PIN",type="password")
    if pin and pin == st.secrets.get("ADMIN_PIN"):
        st.session_state.is_admin = True
        st.success("管理者モード")

# ================================
# ページ管理
# ================================
def nav_button(label, target):
    active = st.session_state.page == target

    st.markdown(
        f'<div class="nav-card {"active" if active else ""}">',
        unsafe_allow_html=True
    )

    clicked = st.button(label,key=f"nav_{target}")
    st.markdown("</div>", unsafe_allow_html=True)
    if clicked and not active:
        st.session_state.page = target
        st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "home"

st.markdown("## 🔽 機能を選択してください")

col1, col2, col3, col4 = st.columns(4)

with col1:
    nav_button("🏠 ホーム", "home")
with col2:
    nav_button("💬 チャット", "chat")
with col3:
    nav_button("🔄 授業変更", "change")
with col4:
    nav_button("📜 履歴", "history")
page = st.session_state.page

# ================================
# ページ：ホーム
# ================================
if page == "home":
    st.markdown("""
    ### ようこそ！

    このアプリは **阿南高専の学生向けサポートAI** です。  
    学校生活でよくある疑問を、AIがすぐに解決します。
    """)

    st.markdown("### 🔍 できること")
    st.markdown("""
    - 💬 **チャット**  
      校則・成績・髪型・進路・奨学金などの質問

    - 🔄 **授業変更**  
      クラスごとの最新の授業変更情報を確認

    - 📜 **履歴**  
      過去の質問と回答を一覧で確認
    """)

    st.markdown("### 🚀 使い方")
    st.markdown("""
    1. 上のメニューから機能を選択  
    2. 質問やクラスを入力  
    3. AIの回答を確認
    """)

    st.info("※ 本アプリは学内向けの試験運用版です。")

# ================================
# ページ：質問
# ================================
elif page == "chat":
    st.write("例: 1年2組 火曜3限 / 髪型の校則は？ / 赤点の基準は？")
    q = st.text_input(
        "",
        placeholder="質問してみましょう",
        label_visibility="collapsed"
    )

    if st.button("送信"):
        if not rate_limit():
            st.stop()

        ok,msg = validate_input(q)
        if not ok:
            st.error(msg)
            st.stop()

        try:
            with st.spinner("考えています..."):
                ans = ask_question(
                    q,
                    dbs["timetable"],
                    dbs["grooming"],
                    dbs["grades"],
                    dbs["abstract"],
                    dbs["cycle"],
                    dbs["abroad"],
                    dbs["sinro"],
                    dbs["part"],
                    dbs["other"],
                    dbs["money"],
                    dbs["domitory"],
                    dbs["clab"],
                )
            
            safe_q = html.escape(q)
            safe_a = html.escape(ans)

            if len(safe_a) > 120:
                with st.expander("回答を表示"):
                    st.write(safe_a)
            else:
                st.success(ans)
            add_history(safe_q,safe_a)
        
        except Exception as e:
            logging.warning(e)
            st.error("内部エラーが発生しました")

# ================================
# ページ：授業変更
# ================================
elif page == "change":
    st.header("授業変更")
    st.write("例：3I 4I などクラスのみで")
    c = st.text_input(
        "",
        placeholder="質問してみましょう",
        label_visibility="collapsed"
    )
    if st.button("取得"):
        result = fetch_class_changes(c if c else None)
        st.info(result)
        add_history(c or "全体", html.escape(result))

# ================================
# ページ：履歴
# ================================
elif page == "history":
    st.header("質問履歴")
    if st.session_state.is_admin:
        if st.button("🗑️ 履歴をすべて削除する"):
            conn = sqlite3.connect("history.db")
            c = conn.cursor()
            c.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            st.rerun()

    history_data = load_history()

    if not history_data:
        st.info("履歴はありません。")
    
    for row in history_data:
        # load_historyのSQL変更に伴い、rowは (id, question, answer, timestamp) になっています
        h_id, q, a, t = row
        t_jp = datetime.fromisoformat(t).strftime("%Y/%m/%d %H:%M")
        
        # コンテナを使ってグループ化
        with st.container():
            # 削除ボタンを右端に配置するためのカラム分割
            col_text, col_btn = st.columns([0.85, 0.15])
            
            with col_text:
                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                            <div style="font-weight: bold; color: #31333F;">📌 {q}</div>
                            <div style="margin-top: 5px; color: #555;">{a}</div>
                            <div style="font-size: 0.8em; color: #888; margin-top: 10px; text-align: right;">{t_jp}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_btn:
                if st.session_state.is_admin:
                    if st.button("削除", key=f"del_{h_id}"):
                        delete_history_item(h_id)
                        st.rerun()