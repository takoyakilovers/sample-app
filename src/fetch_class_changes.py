import requests
from bs4 import BeautifulSoup

PASSWORD = "A7133"
TARGET_URL = "https://www.anan-nct.ac.jp/campuslife/update/"

def fetch_class_changes(target_class=None):
    """
    授業変更ページから変更情報を取得する。
    target_class に '1-2' '1-3' '2M' などが入れば、その行だけ抽出。
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    })


    # Step1: パスワードを送信してログイン
    login_url = "https://www.anan-nct.ac.jp/wp-login.php?action=postpass"
    payload = {"post_password": PASSWORD}
    session.post(login_url, data=payload)

    # Step2: ログイン後のページを取得
    response = session.get(TARGET_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    # 本文を抽出
    body = soup.find("div", class_="entry-body")
    if not body:
        return "授業変更データが取得できませんでした。（構造が変わった可能性）"

    text = body.get_text("\n", strip=True)

    # クラス指定なし → 全部返す
    if not target_class:
        return text

    # クラス指定あり → マッチする行だけ抽出
    results = []
    for line in text.split("\n"):
        if target_class in line:
            results.append(line)

    if not results:
        return f"{target_class} の授業変更情報は見つかりませんでした。"

    return "\n".join(results)