import json
import re
import numpy as np
import torch
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fetch_class_changes import fetch_class_changes
from openai import OpenAI

# ================================
# パス設定（★重要）
# ================================
BASE_DIR = Path(__file__).parent        # src/
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

OPENAI_MODEL_NAME = "openai/gpt-oss-120b"
print(f"--- INFO: LLMモデルを {API_BASE_URL} の {OPENAI_MODEL_NAME} に設定しました。---")

# ================================
# Embeddingモデル
# ================================
embedding_model_name = "intfloat/multilingual-e5-large"
print(f"--- INFO: Embeddingモデル {embedding_model_name} をロード中... ---")
embed_model = SentenceTransformer(embedding_model_name)
print("--- INFO: Embeddingモデルのロード完了 ---")

# ================================
# ファイル読み込み
# ================================
def load_rules_from_file(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"--- INFO: {filepath.name} を読み込みました ---")
        return content
    except FileNotFoundError:
        print(f"⚠ ファイルが見つかりません: {filepath}")
        return ""
    except Exception as e:
        print(f"⚠ 読み込みエラー {filepath}: {e}")
        return ""

# ================================
# 時間割JSON読み込み（★修正済）
# ================================
with open(DATA_DIR / "timetable1.json", "r", encoding="utf-8") as f:
    timetable_data = json.load(f)

# ================================
# 時間割テキスト化
# ================================
def flatten_timetable(data):
    lines = []
    for year, grades in data.items():
        for grade, classes in grades.items():
            for class_name, days in classes.items():
                for day, periods in days.items():
                    for p in periods:
                        lines.append(
                            f"{year}年度 {grade}{class_name} {day}{p['時限']}限:"
                            f"{p['科目']}（{p['教員']}）@{p['教室']}"
                        )
    return "\n".join(lines)

# ================================
# Vector DB 初期化
# ================================
def initialize_vector_db(text: str):
    if not text:
        return None

    chunks = [t.strip() for t in text.split("\n\n") if t.strip()]
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    return list(zip(chunks, embeddings))

# ================================
# RAG検索
# ================================
def get_rule_context_from_rag(query: str, db: list, k: int = 5):
    if not db:
        return None, None

    q_vec = embed_model.encode(query)
    vectors = np.array([v for _, v in db])
    texts = [t for t, _ in db]

    sims = cosine_similarity(q_vec.reshape(1, -1), vectors)[0]
    idxs = np.argsort(sims)[-k:][::-1]

    context = "\n---\n".join(texts[i] for i in idxs)
    question = f"次の校則データをもとに質問に答えてください。\n質問: {query}"
    return context, question

# ================================
# 正規化
# ================================
def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("　", " ").replace("ー", "-")
    return text

# ================================
# クラス・曜日・時限抽出
# ================================
def detect_class_from_query(q):
    q = normalize(q)
    m = re.search(r"1[- ]?([1-4])", q)
    if m:
        return ("1年", f"{m.group(1)}組")
    m = re.search(r"([2-5])([meicz])", q)
    if m:
        return (f"{m.group(1)}年", m.group(2).upper())
    return None

def detect_day_from_query(q):
    for d in ["月", "火", "水", "木", "金"]:
        if d in q:
            return d + "曜"
    return None

def detect_period_from_query(q):
    m = re.search(r"([1-6])限", q)
    return int(m.group(1)) if m else None

# ================================
# 時間割取得
# ================================
def get_relevant_text(data, year, grade, cls, day, period=None):
    try:
        schedule = data[year][grade][cls][day]
    except KeyError:
        return None

    out = []
    for p in schedule:
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
    if any(k in q for k in ["何限", "授業", "時間割"]):
        return "timetable"
    if any(k in q for k in ["髪", "服装", "身だしなみ"]):
        return "grooming"
    if any(k in q for k in ["成績", "赤点"]):
        return "grades"
    if any(k in q for k in ["欠席"]):
        return "abstract"
    if any(k in q for k in ["自転車"]):
        return "cycle"
    if any(k in q for k in ["留学"]):
        return "abroad"
    if any(k in q for k in ["進路"]):
        return "sinro"
    if any(k in q for k in ["バイト"]):
        return "part"
    if any(k in q for k in ["奨学金"]):
        return "money"
    if any(k in q for k in ["寮"]):
        return "domitory"
    if any(k in q for k in ["部活"]):
        return "clab"
    return "other"

# ================================
# メインQA関数
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

    if intent == "timetable":
        cls = detect_class_from_query(query)
        day = detect_day_from_query(query) or "月曜"
        period = detect_period_from_query(query)

        if not cls:
            return "クラスが特定できませんでした。"

        grade, cname = cls
        context = get_relevant_text(
            timetable, "2025", grade, cname, day, period
        )

        if not context:
            return "時間割が見つかりませんでした。"

        prompt = f"以下の時間割をもとに質問に答えてください。\n{context}"
        max_tokens = 300

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
        db = db_map[intent]
        context, question = get_rule_context_from_rag(query, db)
        if not context:
            return "該当する情報が見つかりませんでした。"

        prompt = f"{context}\n\n{question}"
        max_tokens = 500

    try:
        res = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "AIとの通信中にエラーが発生しました。"
