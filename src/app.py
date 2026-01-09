import streamlit as st
from pathlib import Path

from anan_ai import ask_question, initialize_vector_db
from fetch_class_changes import fetch_class_changes
from history import save_history, load_history, clear_history

# ================================
# ページ設定
# ================================
st.set_page_config(
    page_title="阿南高専 AI アシスタント",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 阿南高専 AI アシスタント")

# ================================
# パス設定
# ================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ================================
# ルールファイル読み込み
# ================================
def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")

# ================================
# 校則 Vector DB（キャッシュ）
# ================================
@st.cache_resource
def load_all_rule_vectors():
    return {
        "grooming": initialize_vector_db(load_text(DATA_DIR / "grooming.txt")),
        "grades": initialize_vector_db(load_text(DATA_DIR / "grades.txt")),
        "abstract": initialize_vector_db(load_text(DATA_DIR / "abstract.txt")),
        "cycle": initialize_vector_db(load_text(DATA_DIR / "cycle.txt")),
        "abroad": initialize_vector_db(load_text(DATA_DIR / "abroad.txt")),
        "sinro": initialize_vector_db(load_text(DATA_DIR / "sinro.txt")),
        "part": initialize_vector_db(load_text(DATA_DIR / "part.txt")),
        "other": initialize_vector_db(load_text(DATA_DIR / "other.txt")),
        "money": initialize_vector_db(load_text(DATA_DIR / "money.txt")),
        "domitory": initialize_vector_db(load_text(DATA_DIR / "domitory.txt")),
        "clab": initialize_vector_db(load_text(DATA_DIR / "clab.txt")),
    }

rule_vectors = load_all_rule_vectors()

# ================================
# サイドバー
# ================================
st.sidebar.header("📋 メニュー")

menu = st.sidebar.radio(
    "選択してください",
    ["AIに質問", "授業変更情報", "質問履歴"]
)

if st.sidebar.button("🗑 履歴を削除"):
    clear_history()
    st.sidebar.success("履歴を削除しました")

# ================================
# AIに質問
# ================================
if menu == "AIに質問":
    st.subheader("💬 質問入力")

    user_input = st.text_input(
        "質問を入力してください",
        placeholder="例：1-3の月曜2限は何の授業？"
    )

    if st.button("送信") and user_input:
        with st.spinner("AIが考えています..."):
            answer = ask_question(
                query=user_input,
                db_map=rule_vectors
            )

        st.markdown("### 🤖 回答")
        st.write(answer)

        save_history(user_input, answer)

# ================================
# 授業変更情報
# ================================
elif menu == "授業変更情報":
    st.subheader("📢 授業変更情報")

    with st.spinner("取得中..."):
        changes = fetch_class_changes()

    if not changes:
        st.info("現在、授業変更はありません。")
    else:
        for c in changes:
            st.markdown(
                f"""
                **📅 {c['date']}**  
                {c['content']}
                """
            )

# ================================
# 履歴
# ================================
elif menu == "質問履歴":
    st.subheader("🕘 質問履歴")

    history = load_history()

    if not history:
        st.info("履歴はありません。")
    else:
        for h in history:
            st.markdown(
                f"""
                **🕒 {h['time']}**  
                **Q:** {h['question']}  
                **A:** {h['answer']}
                ---
                """
            )
