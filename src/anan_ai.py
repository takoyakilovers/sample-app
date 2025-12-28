import json
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fetch_class_changes import fetch_class_changes
from openai import OpenAI

# ターゲットとなるOpenAI互換APIのエンドポイントとキー
# 提供されたコードを基に設定します。
API_BASE_URL = "http://hpc04.anan-nct.ac.jp:8000/v1"
API_KEY = "EMPTY" # APIキーが不要な場合

# クライアントをグローバルに初期化
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# 使用するモデル名 (サーバー側で提供されているものに合わせる)
OPENAI_MODEL_NAME = "openai/gpt-oss-120b"
print(f"--- INFO: LLMモデルを {API_BASE_URL} の {OPENAI_MODEL_NAME} に設定しました。---")

# Embeddingモデルの準備はそのまま維持します (RAG用)
embedding_model_name = "intfloat/multilingual-e5-large"
print(f"--- INFO: Embeddingモデル {embedding_model_name} をロード中... ---")
embed_model = SentenceTransformer(embedding_model_name)
print("--- INFO: Embeddingモデルのロード完了 ---")

# ==== ファイル読み込み関数 (拡張) ====
def load_rules_from_file(filepath: str) -> str:
    """
    指定されたパスから校則テキストを読み込む。
    読み込みエラーが発生した場合は、空文字列を返す。
    """
    try:
        # 実際の実行環境に合わせてファイルパスとエンコーディングを調整してください
        with open(filepath, "r", encoding="utf-8") as f:
            rules_content = f.read()
        print(f"--- INFO: ファイル '{filepath}' を読み込みました。---")
        return rules_content
    except FileNotFoundError:
        print(f"エラー: 校則ファイル '{filepath}' が見つかりません。このDBは無効化されます。")
        return ""
    except Exception as e:
        print(f"エラー: ファイル読み込み中にエラーが発生しました: {e}")
        return ""

# ==== JSONを読み込む ====
# 実際の実行環境に合わせてファイルパスを調整してください
with open("data/timetable1.json", "r", encoding="utf-8") as f:
    timetable_data = json.load(f)

# ==== JSONをテキスト化（RAG的前処理）====
def flatten_timetable(data):
    # (既存の関数。時間割データをフラットなテキストにする)
    text_data = []
    for year, grades in data.items():
        for grade, classes in grades.items():
            for class_name, days in classes.items():
                for day, periods in days.items():
                    for period in periods:
                        line = f"{year}年度 {grade}{class_name} {day}{period['時限']}限: {period['科目']}（{period['教員']}）@{period['教室']}"
                        text_data.append(line)
    return "\n".join(text_data)

knowledge_text = flatten_timetable(timetable_data)

# ==== RAG用 ベクトルDB初期化関数 (変更なし) ====
def initialize_vector_db(text: str):
    """校則テキストをチャンク化し、ベクトル化して、DB（リスト）を作成する"""
    if not text:
        return None

    # チャンク化：ここでは「空行」で区切って条文単位に分割
    chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
    if not chunks:
        print("警告: テキストから有効なチャンクが抽出できませんでした。")
        return None

    # ベクトル化
    embeddings = embed_model.encode(chunks, show_progress_bar=False)

    # DBの構造： [(条文テキスト, ベクトル numpy array), ...] のリストとして格納
    vector_db = list(zip(chunks, embeddings))

    print(f"--- INFO: ベクトルDBの初期化完了。チャンク数: {len(chunks)} ---")
    return vector_db


# ==== RAG用 コンテキスト取得関数 (変更なし) ====
def get_rule_context_from_rag(query: str, rule_vector_db: list, k: int = 5):
    """
    質問をベクトル化し、ルールDBから最も関連性の高い条文を検索して返す
    """
    if not rule_vector_db:
        return None, f"ユーザーの質問「{query}」に対する回答を生成できませんでした。"

    # 1. 質問をベクトル化
    query_vector = embed_model.encode(query)

    # 2. 検索用のデータ準備
    chunks = [item[0] for item in rule_vector_db]
    vectors = np.array([item[1] for item in rule_vector_db])

    # 3. 類似度計算 (コサイン類似度を使用)
    # query_vectorを (1, embed_dim) の形に整形して計算
    similarities = cosine_similarity(query_vector.reshape(1, -1), vectors)

    # 4. Top k個のインデックスを取得 (類似度が高い順)
    top_k_indices = np.argsort(similarities[0])[-k:][::-1]

    # 5. 関連条文（チャンク)を結合してコンテキストとする
    context = "\n---\n".join([chunks[i] for i in top_k_indices])

    question_text = f"ユーザーの質問「{query}」に対する回答を、以下の【校則データ】に基づいて生成してください。"

    return context, question_text

# ==== 表記ゆれ正規化関数 (変更なし) ====
def normalize(text: str) -> str:
    # 漢数字、全角数字、スペースなどを半角・統一形式に変換
    text = text.replace("一", "1").replace("二", "2").replace("三", "3").replace("四", "4")
    text = text.replace("０", "0").replace("１", "1").replace("２", "2").replace("３", "3").replace("４", "4").replace("５", "5").replace("６", "6")
    text = text.replace("　", " ").replace("ー", "-").replace("組", "組")
    return text.lower()

# ==== クラスの抽出 (変更なし) ====
def detect_class_from_query(query: str):
    query_n = normalize(query)
    # 1年生（1-1 ～ 1-4）のパターン
    m = re.search(r"1\s*[-]\s*([1-4])", query_n)
    if m: return ("1年", f"{m.group(1)}組")
    m = re.search(r"1\s*年\s*([1-4])\s*組", query_n)
    if m: return ("1年", f"{m.group(1)}組")
    m = re.search(r"([1-4])\s*組", query_n)
    if m: return ("1年", f"{m.group(1)}組")
    # 2年生以上（2M, 3E など）のパターン
    m = re.search(r"([2-5])\s*([meicz])", query_n, re.IGNORECASE)
    if m: return (f"{m.group(1)}年", m.group(2).upper())
    return None

# ==== 曜日の抽出 (変更なし) ====
def detect_day_from_query(query: str) -> str | None:
    query_n = normalize(query)
    if "月" in query_n: return "月曜"
    if "火" in query_n: return "火曜"
    if "水" in query_n: return "水曜"
    if "木" in query_n: return "木曜"
    if "金" in query_n: return "金曜"
    return None

# ==== 時限の抽出 (変更なし) ====
def detect_period_from_query(query: str) -> int | None:
    query_n = normalize(query)
    m = re.search(r"([1-6])\s*[限時]", query_n)
    if m: return int(m.group(1))
    return None

# ==== サンプルの時間割抽出 (変更なし) ====
def get_relevant_text(data, year="2025", grade="1年", class_name="2組", day="月曜", period=None):
    try:
        day_schedule = data[year][grade][class_name][day]
    except KeyError:
        return None
    lines = []
    if period:
        for p in day_schedule:
            if p['時限'] == period:
                line = f"{day}{period}限: {p['科目']}（{p['教員']}）@{p['教室']}"
                lines.append(line)
                break
    else:
        for p in day_schedule:
            if p['時限'] in [1, 2, 3, 4]:
                line = f"{day}{p['時限']}限: {p['科目']}（{p['教員']}）@{p['教室']}"
                lines.append(line)
    return "\n".join(lines) if lines else None

# ==== 質問の意図判定関数 (変更なし) ====
def determine_intent(query: str):
    """
    質問が時間割、身だしなみ、成績表、特別欠席のいずれに関するものかを判定する
    """
    query_n = normalize(query)

    # 1. 時間割のキーワード
    if any(k in query_n for k in ["時間割", "何組", "何限", "教室", "授業", "今日", "限"]):
        return "timetable"

    # 2. 身だしなみのキーワード (style.txt)
    if any(k in query_n for k in ["身だしなみ", "髪", "服装", "制服", "略装", "靴", "アクセサリー"]):
        return "grooming"

    # 3. 成績表のキーワード (grade.txt)
    if any(k in query_n for k in ["成績", "成績表", "単位", "点数", "GPA", "評価", "赤点", "原点", "何点"]):
        return "grades"

    # 4. 特別欠席のキーワード (absent.txt)
    if any(k in query_n for k in ["欠席", "欠課", "ストライキ", "交通機関", "汽車", "病気", "インフル", "特別"]):
        return "abstract"

    # 5. 自転車に関するキーワード
    if any(k in query_n for k in ["自転車", "駐輪場", "バイク", "通学", "原付", "二輪車"]):
        return "cycle"

    # 6. 留学に関するキーワード
    if any(k in query_n for k in ["留学", "海外", "研修", "台湾", "ニュージーランド", "インターンシップ"]):
        return "abroad"

    # 7. 進路に関するキーワード (sinro.txt)
    if any(k in query_n for k in ["進路", "就職", "進学", "大学", "専攻科", "推薦", "求人", "企業","編入"]):
        return "sinro"

    # 8. アルバイト/課外活動に関するキーワード (part.txt)
    if any(k in query_n for k in ["アルバイト", "バイト"]):
        return "part"

    # 9. その他に関するキーワード (other.txt)
    if any(k in query_n for k in ["校則", "規則", "携帯電話", "スマホ", "スマートフォン", "いじめ", "始業時間", "授業時間", "コース配属", "保護者面談", "高専", "5年一貫"]):
        return "other"

    # 10. 奨学金/学費に関するキーワード (money.txt)
    if any(k in query_n for k in ["奨学金", "学費", "授業料", "免除", "お金", "費用", "振込"]):
        return "money"

    # 11. 寮生活に関するキーワード (domitory.txt)
    if any(k in query_n for k in ["寮", "寮生活", "阿南寮", "門限", "外泊", "帰省", "部屋"]):
        return "domitory"

    # 12. 部活動に関するキーワード (clab.txt)
    if any(k in query_n for k in ["部活", "部活動", "クラブ", "サークル", "大会", "兼部"]):
        return "clab"

    return "general"

# ==== LLMに質問（OpenAI API版） ====
# 呼び出し側の引数に合わせて、全てのDB変数を引数として受け取るように修正
def ask_question(query, timetable_data, grooming_db, grades_db, abstract_db, cycle_db, abroad_db, sinro_db, part_db, other_db, money_db, domitory_db, clab_db):

    # 意図判定
    intent = determine_intent(query)

    context = None
    question_text = ""
    prompt_type = ""
    db = None
    db_name = None # RAGで使うデータベース名を格納するための変数

    # 1. RAG対象データベースの割り当てと名前の設定
    if intent == "grooming":
        db = grooming_db
        db_name = "身だしなみ校則"
    elif intent == "grades":
        db = grades_db
        db_name = "成績表ルール"
    elif intent == "abstract":
        db = abstract_db
        db_name = "特別欠席ルール"
    elif intent == "cycle":
        db = cycle_db
        db_name = "自転車規則"
    elif intent == "abroad":
        db = abroad_db
        db_name = "留学・海外研修"
    elif intent == "sinro":
        db = sinro_db
        db_name = "進路"
    elif intent == "part":
        db = part_db
        db_name = "アルバイト・課外活動"
    elif intent == "other":
        db = other_db
        db_name = "その他規則"
    elif intent == "money":
        db = money_db
        db_name = "奨学金・学費"
    elif intent == "domitory":
        db = domitory_db
        db_name = "寮生活"
    elif intent == "clab":
        db = clab_db
        db_name = "部活動"
    # timetableまたはgeneralの場合は db は None のまま

    # 2. 意図に応じたデータの取得 (時間割 または RAG)
    if intent == "timetable":
        class_info = detect_class_from_query(query)
        day = detect_day_from_query(query)
        period = detect_period_from_query(query)

        if not class_info:
            return "クラスを特定できませんでした。例: 1年2組、1-2、二組 など"

        grade, class_name = class_info
        day = day or "月曜"

        context = get_relevant_text(timetable_data, year="2025", grade=grade, class_name=class_name, day=day, period=period)

        if not context:
            return f"{grade}{class_name}の{day}の時間割が見つかりませんでした。"

        if period:
            question_text = f"{grade}{class_name}の{day}{period}限の授業は何ですか?"
        else:
            question_text = f"{grade}{class_name}の{day}の時間割を教えてください。"

        prompt_type = "timetable"

    elif db is not None:
        # RAGを使用するDBの処理を共通化
        if not db:
            # db変数はNoneでないが、中身が空の場合（ファイル読み込み失敗時など）
            return f"{db_name}データベースがロードされていません。"

        context, question_text = get_rule_context_from_rag(query, db)

        if not context:
            # RAG検索しても関連情報が見つからなかった場合
            return f"{db_name}に関する情報がデータに見つかりませんでした。"

        prompt_type = "rules"

    else:
        # 質問の意図のリストを更新
        return "質問の意図（時間割、身だしなみ、成績、欠席、留学、進路、アルバイト、寮、部活など）を特定できませんでした。詳しく質問してください。"


    # == プロンプトテンプレートの切り替え ==
    if prompt_type == "timetable":
        # 時間割用：フレンドリーで簡潔な回答を促すプロンプト
        prompt = f"""あなたは阿南高専の学生サポートAIです。
以下の時間割データを使って、自然な口調で答えてください。

【時間割データ】
{context}

【質問】
{question_text}

【回答の指示】
- すべての授業情報（時限、科目名、先生、教室）を含めて答える
- 簡潔に2〜3文程度でまとめる
- 情報を省略せず、でも読みやすくまとめる
- 必要な情報のみ回答する
【回答】
"""
        max_tokens = 300
    elif prompt_type == "rules":
        # 校則用：丁寧だけど親しみやすい回答を促すプロンプト
        prompt = f"""あなたは阿南高専の学生サポートAIです。
以下の参照データを使って、自然な口調で答えてください。

【参照データ】
{context}

【質問】
{question_text}

【回答の指示】
- 難しい言葉遣いは避け、わかりやすく説明する
- 必要な情報は正確に伝えつつ、会話的に答える
- 表形式（テーブル、|記号）は使わず、文章で答える
- データにない情報は「それについては情報がありません」と答える
- **回答は必要な情報をすべて含め、途中で終わらせずに完結させる** # ← 新たに追加

【回答】
"""
        max_tokens = 500
    else:
        # ここには到達しないはずだが、念のため
        return "内部エラー: プロンプトタイプが不明です。"

    # === LLM実行 ===
    # clientはグローバル変数として定義されていることを想定
    # OPENAI_MODEL_NAMEはグローバル変数として定義されていることを想定
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7  # 少し高めにして自然な口調を促す
        )
        response_text = resp.choices[0].message.content

    except Exception as e:
        # エラー発生時の処理（メインループで囲まれたとき、この print は表示されない可能性あり）
        # print(f"エラー: OpenAI API呼び出し中にエラーが発生しました: {e}")
        return "AIモデルへの問い合わせ中にエラーが発生しました。"

    # === 回答の後処理 ===
    if response_text is None:
        return "AIモデルが回答を生成できませんでした。"

    # プロンプトより後だけ切り抜く
    answer = response_text.split("【回答】")[-1].strip()

    # 不要なプレアンブルを削除 (汎用的な前処理)
    answer = re.sub(
        r'^(回答「.*?」|【校則データ】|【参照データ】|【時間割データ】|あなたは|この度は|さて|実はこの件に関しては|なぜなら|しかし|一般的に|一般的には|そこで|まず|承知いたしました|回答は次のとおりです|回答は以下のとおりです)[^\n]*\n?',
        '',
        answer,
        flags=re.MULTILINE | re.DOTALL
    ).strip()

    # 校則用（ルール）の後処理
    if prompt_type == "rules":
        answer = re.sub(
            r'^\(.*\)\s*\n?第\s*\d+\s*条.*|^\-{3,}.*$',
            '',
            answer,
            flags=re.MULTILINE | re.DOTALL
        ).strip()
        answer = re.sub(r'\*+\d+', '', answer)
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = re.sub(r'以下の【参照データ】.*', '', answer, flags=re.DOTALL) # 誤って含んだ場合を削除

    # 時間割用の後処理（緩和版）
    if prompt_type == "timetable":
        # 【】タグだけ削除し、文章の流れは保持
        answer = re.sub(r'【[^】]*】', '', answer)

    # 最終的な空行削除と整形
    answer = '\n'.join([line.strip() for line in answer.split('\n') if line.strip()])

    return answer