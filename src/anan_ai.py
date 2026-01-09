import json
import re
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ================================
# パス設定
# ================================
BASE_DIR = Path(__file__).parent
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

# ================================
# Embedding（軽量・必須）
# ================================
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ================================
# 時間割JSON
# ================================
TIMETABLE_DATA = {}
timetable_path = DATA_DIR / "timetable1.json"

if timetable_path.exists():
    with open(timetable_path, "r", encoding="utf-8") as f:
        TIMETABLE_DATA = json.load(f)

# ================================
# Vector DB
# ================================
def initialize_vector_db(text: str):
    if not text:
        return []
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    vectors = embed_model.encode(chunks, show_progress_bar=False)
    return list(zip(chunks, vectors))

def get_rule_context(query: str, db: list, k: int = 5):
    if not db:
        return None

    q_vec = embed_model.encode(query)
    vecs = np.array([v for _, v in db])
    texts = [t for t, _ in db]

    sims = cosine_similarity(q_vec.reshape(1, -1), vecs)[0]
    idxs = np.argsort(sims)[-k:][::-1]

    return "\n---\n".join(texts[i] for i in idxs)

# ================================
# 判定系
# ================================
def normalize(text: str):
    return text.lower().replace("　", " ").replace("ー", "-")

def detect_class(q):
    q = normalize(q)
    m = re.search(r"1[- ]?([1-4])", q)
    if m:
        return ("1年", f"{m.group(1)}組")
    m = re.search(r"([2-5])([meicz])", q)
    if m:
        return (f"{m.group(1)}年", m.group(2).upper())
    return None

def detect_day(q):
    for d in ["月", "火", "水", "木", "金"]:
        if d in q:
            return d + "曜"
    return "月曜"

def detect_period(q):
    m = re.search(r"([1-6])限", q)
    return int(m.group(1)) if m else None

def determine_intent(q):
    q = normalize(q)
    if "時間割" in q or "授業" in q:
        return "timetable"
    if "髪" in q or "服装" in q:
        return "grooming"
    if "成績" in q:
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
# ★ メイン関数
# ================================
def ask_question(query: str, db_map: dict):
    intent = determine_intent(query)

    # ---- 時間割 ----
    if intent == "timetable":
        cls = detect_class(query)
        if not cls:
            return "クラスが特定できませんでした。"

        grade, cname = cls
        day = detect_day(query)
        period = detect_period(query)

        try:
            day_data = TIMETABLE_DATA["2025"][grade][cname][day]
        except Exception:
            return "時間割が見つかりませんでした。"

        lines = [
            f"{day}{p['時限']}限: {p['科目']}（{p['教員']}）@{p['教室']}"
            for p in day_data
            if not period or p["時限"] == period
        ]

        prompt = "\n".join(lines)

    # ---- 校則系 ----
    else:
        db = db_map.get(intent)
        context = get_rule_context(query, db)
        if not context:
            return "該当する情報が見つかりませんでした。"

        prompt = f"{context}\n\n質問: {query}"

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("LLM ERROR:", e)
        return "AIとの通信中にエラーが発生しました。"

__all__ = ["ask_question", "initialize_vector_db"]
