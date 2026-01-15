import os
import requests
from bs4 import BeautifulSoup

# ================================
# 環境変数（絶対に直書きしない）
# ================================
PASSWORD = os.getenv("ANAN_UPDATE_PASSWORD")
TARGET_URL = "https://www.anan-nct.ac.jp/campuslife/update/"
LOGIN_URL = "https://www.anan-nct.ac.jp/wp-login.php?action=postpass"

if not PASSWORD:
    raise RuntimeError("ANAN_UPDATE_PASSWORD is not set")

# ================================
# 授業変更取得
# ================================
def fetch_class_changes(target_class: str | None = None) -> str:
    """
    授業変更ページから変更情報を取得
    戻り値: 表示用文字列
    """

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    try:
        # パスワードログイン
        session.post(
            LOGIN_URL,
            data={"post_password": PASSWORD},
            timeout=10
        )

        # ページ取得
        res = session.get(TARGET_URL, timeout=10)
        res.raise_for_status()

    except requests.RequestException:
        return "授業変更ページに接続できませんでした。"

    soup = BeautifulSoup(res.text, "html.parser")

    body = soup.find("div", class_="entry-body")
    if not body:
        return "授業変更情報が見つかりませんでした。"

    lines = body.get_text("\n", strip=True).split("\n")

    results = []
    for line in lines:
        if not line.strip():
            continue
        if target_class and target_class not in line:
            continue
        results.append(line)

    if not results:
        if target_class:
            return f"{target_class} の授業変更はありません。"
        return "現在、授業変更はありません。"

    header = "📢 授業変更情報\n"
    content = "\n".join(f"・{r}" for r in results)

    return f"{header}\n{content}"
