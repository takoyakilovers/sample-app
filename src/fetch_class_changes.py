import requests
from bs4 import BeautifulSoup

PASSWORD = "A7133"
TARGET_URL = "https://www.anan-nct.ac.jp/campuslife/update/"

def fetch_class_changes(target_class=None):
    """
    授業変更ページから変更情報を取得
    戻り値: [{date: str, content: str}, ...]
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    # パスワードログイン
    login_url = "https://www.anan-nct.ac.jp/wp-login.php?action=postpass"
    session.post(login_url, data={"post_password": PASSWORD})

    # ページ取得
    res = session.get(TARGET_URL, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    body = soup.find("div", class_="entry-body")
    if not body:
        return []

    lines = body.get_text("\n", strip=True).split("\n")

    results = []
    for line in lines:
        if target_class and target_class not in line:
            continue

        results.append({
            "date": "最新",      # 日付が無いので固定
            "content": line
        })

    return results
