import streamlit as st
import json
import sqlite3
from datetime import datetime
from anan_ai import ask_question, load_rules_from_file, initialize_vector_db
from fetch_class_changes import fetch_class_changes
from history import save_history, load_history, clear_history
from pathlib import Path
BASE_DIR = Path(__file__).parent   # src/


# ================================
# CSS
# ================================
def load_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# set_page_config は必ず一番最初に呼ぶ必要があります
st.set_page_config(page_title="阿南高専 chatbot", page_icon="⏰")
load_css()

st.title("🏫 阿南高専 学生サポートAI")
st.info("""
このAIでは以下の質問ができます : 
・時間割の確認
・校則 (髪型・成績・欠席など)
・奨学金・寮・部活動
・授業変更情報
""")

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
# データ読み込み
# ================================
@st.cache_resource
def load_all_data():
    data_dir = BASE_DIR / "data"

    with open(data_dir / "timetable1.json", "r", encoding="utf-8") as f:
        timetable = json.load(f)

    def load_db(path):
        return initialize_vector_db(load_rules_from_file(path))

    return {
        "timetable": timetable,
        "grooming": load_db(data_dir / "style.txt"),
        "grades": load_db(data_dir / "grade.txt"),
        "abstract": load_db(data_dir / "abstract.txt"),
        "cycle": load_db(data_dir / "cycle.txt"),
        "abroad": load_db(data_dir / "abroad.txt"),
        "sinro": load_db(data_dir / "sinro.txt"),
        "part": load_db(data_dir / "part.txt"),
        "other": load_db(data_dir / "other.txt"),
        "money": load_db(data_dir / "money.txt"),
        "domitory": load_db(data_dir / "domitory.txt"),
        "clab": load_db(data_dir / "clab.txt"),
    }


dbs = load_all_data()

# ================================
# ページ管理
# ================================
with st.sidebar:
    st.markdown("### 📌 メニュー")
    page = st.radio(
        "ページを選択",
        ["💬 質問", "🔄 授業変更", "📜 履歴"],
        label_visibility="collapsed"
    )
if page == "💬 質問":
    page = "chat"
elif page == "🔄 授業変更":
    page = "change"
elif page == "📜 履歴":
    page = "history"

# ================================
# ページ：質問
# ================================
if page == "chat":
    st.write("例: 1年2組 火曜3限 / 髪型の校則は？ / 赤点の基準は？")
    q = st.text_input(
        "質問を入力",
        placeholder = "例:1年2組 火曜3限 / 髪型の校則は? / 赤点の基準は?"
    )

    if st.button("送信") or q and st.session_state.get("enter_pressed"):
        if q.strip():
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
            if len(ans) > 120:
                with st.expander("回答を表示"):
                    st.write(ans)
            else:
                st.success(ans)
            add_history(q,ans)

# ================================
# ページ：授業変更
# ================================
elif page == "change":
    st.header("授業変更")
    c = st.text_input("クラス（例: 1-2, 2M）")
    if st.button("取得"):
        result = fetch_class_changes(c if c else None)
        st.info(result)
        add_history(c or "全体", result)

# ================================
# ページ：履歴
# ================================
elif page == "history":
    st.header("質問履歴")
    # 全削除ボタン（オプション）
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
                # 削除ボタン。keyにIDを含めることで一意にします
                if st.button("削除", key=f"del_{h_id}"):
                    delete_history_item(h_id)
                    st.rerun()