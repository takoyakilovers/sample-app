import json
import re
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ================================
# パス設定（Streamlit Cloud対応）
# ================================
BASE_DIR = Path(__file__).parent          # src/
DATA_DIR = BASE_DIR / "data"

# ================================
# OpenAI互換API設定
# ================================
API_BASE_URL = "http://hpc04.anan-nct.ac.jp:8000/v1"
API_KEY = os.getenv("API_KEY", "EMPTY")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

MODEL_NAME = "openai/gpt-oss-120b"
print(f"--- INFO: LLMモデル {MODEL_NAME} を使用 ---")

# ================================
# Embedding モデル
# ================================
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
print(f"--- INFO: Embeddingモデル {EMBED_MODEL_NAME} をロード中 ---")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("--- INFO: Embeddingモデル ロード完了 ---")

# ================================
# ファイル読み込み
# ================================
def load_rules_from_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠ 規則ファイルが見つかりません: {path}")
        return ""

# ================================
# 時間割JSON読み込み（★重要）
# ================================
timetable_path = DATA_DIR / "timetable1.json"
if timetable_path.exists():
    with open(timetable_path, "r", encoding="utf-8") as f:
        TIMETABLE_DATA = json.load(f)
else:
    print("⚠ timetable1.json が存在しません")
    TIMETABLE_DATA = {}

# ================================
# Vector DB 初期化
# ================================
def initialize_vector_db(text: str):
    if not text:
        return None
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    vectors = embed_model.encode(chunks, show_progress_bar=False)
    return list(zip(chunks, vectors))

# ================================
# RAG検索
# ================================
def get_rule_context(query: str, db: list, k: int = 5):
    if not db:
        return None

    q_vec = embed_model.encode(query)
    texts = [t for t, _ in db]
    vecs = np.array([v for _, v in db])

    sims = cosine_similarity(q_vec.reshape(1, -1), vecs)[0]
    idxs = np.argsort(sims)[-k:][::-1]

    return "\n---\n".join(texts[i] for i in idxs)

# ================================
# 正規化
# ================================
def normalize(text: str) -> str:
    return text.lower().replace("　", " ").replace("ー", "-")

# ================================
# クラス・曜日・時限抽出
# ================================
def detect_class(query):
    q = normalize(query)
    m = re.search(r"1[- ]?([1-4])", q)
    if m:
        return ("1年", f"{m.group(1)}組")
    m = re.search(r"([2-5])([meicz])", q)
    if m:
        return (f"{m.group(1)}年", m.group(2).upper())
    return None

def detect_day(query):
    for d in ["月", "火", "水", "木", "金"]:
        if d in query:
            return d + "曜"
    return "月曜"

def detect_period(query):
    m = re.search(r"([1-6])限", query)
    return int(m.group(1)) if m else None

# ================================
# 時間割取得
# ================================
def get_timetable_text(year, grade, cls, day, period=None):
    try:
        day_data = TIMETABLE_DATA[year][grade][cls][day]
    except Exception:
        return None

    out = []
    for p in day_data:
        if not period or p["時限"] == period:
            out.append(
                f"{day}{p['時限']}限: {p['科目']}（{p['教員']}）@{p['教室']}"
            )
    return "\n".join(out)

# ================================
# 意図判定
# ================================
def determine_intent(q):
    q = normalize(q)
    if any(k in q for k in ["時間割", "授業", "何限"]):
        return "timetable"
    if any(k in q for k in ["髪", "身だしなみ", "服装"]):
        return "grooming"
    if any(k in q for k in ["成績", "赤点"]):
        return "grades"
    if "欠席" in q:
        return "abstract"
    if "自転車" in q:
        return "cycle"
    if "留学" in q:
        return "abroad"
    if "進路" in q:
        return "sinro"
    if "バイト" in q:
        return "part"
    if "奨学金" in q:
        return "money"
    if "寮" in q:
        return "domitory"
    if "部活" in q:
        return "clab"
    return "other"

# ================================
# ★ メイン関数（Import対象）
# ================================
def ask_question(
    query,
    timetable,
    grooming,
    grades,
    abstract,
    cycle,
    abroad,
    sinro,
    part,
    other,
    money,
    domitory,
    clab
):
    intent = determine_intent(query)

    # ---- 時間割 ----
    if intent == "timetable":
        cls = detect_class(query)
        if not cls:
            return "クラスが特定できませんでした。"

        grade, cname = cls
        day = detect_day(query)
        period = detect_period(query)

        text = get_timetable_text("2025", grade, cname, day, period)
        if not text:
            return "時間割が見つかりませんでした。"

        prompt = f"以下の時間割をもとに質問に答えてください。\n{text}"

    # ---- 校則系（RAG）----
    else:
        db_map = {
            "grooming": grooming,
            "grades": grades,
            "abstract": abstract,
            "cycle": cycle,
            "abroad": abroad,
            "sinro": sinro,
            "part": part,
            "other": other,
            "money": money,
            "domitory": domitory,
            "clab": clab,
        }

        context = get_rule_context(query, db_map.get(intent))
        if not context:
            return "該当する情報が見つかりませんでした。"

        prompt = f"{context}\n\n質問: {query}"

    # ---- LLM呼び出し ----
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("LLM ERROR:", e)
        return "AIとの通信中にエラーが発生しました。"
