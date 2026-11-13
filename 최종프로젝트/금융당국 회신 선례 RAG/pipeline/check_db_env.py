"""RAG_DSN 점검: 형식, 비밀번호 입력 상태, 포트, 실제 접속. 비밀번호 값은 어떤 경우에도 출력하지 않는다.
사용: python check_db_env.py
"""
import socket
from urllib.parse import unquote, urlsplit

import psycopg
from dotenv import dotenv_values

from common import ROOT

PLACEHOLDERS = ("사용자", "비밀번호", "DB이름", "여기에_붙여넣기")
UNSAFE = "@:/?#[]% "


def check(dsn):
    if not dsn:
        print("[확인 필요] .env에 RAG_DSN 값이 없습니다.")
        return False
    try:
        u = urlsplit(dsn)
        port = u.port
    except ValueError:
        print("[확인 필요] 주소를 읽지 못했습니다. 비밀번호에 @ : / 같은 문자가 있는지 확인하세요.")
        return False
    raw_pw = u.password or ""
    pw = unquote(raw_pw)
    db = u.path.lstrip("/")
    print("종류:", u.scheme or "(없음)")
    print("사용자:", u.username or "(없음)")
    print("호스트:", u.hostname or "(없음)")
    print("포트:", port or "(없음)")
    print("DB 이름:", db or "(없음)")
    print("비밀번호:", f"입력됨, {len(pw)}자" if pw else "없음")

    ok = True
    if u.scheme not in ("postgresql", "postgres"):
        print("[확인 필요] postgresql:// 로 시작해야 합니다.")
        ok = False
    if not (u.username and pw and u.hostname and port and db):
        print("[확인 필요] 사용자, 비밀번호, 호스트, 포트, DB 이름 중 빠진 값이 있습니다.")
        ok = False
    if any(p in dsn for p in PLACEHOLDERS):
        print("[확인 필요] 예시 글자가 그대로 남아 있습니다.")
        ok = False
    if pw and len(pw) < 16:
        print("[참고] 비밀번호가 16자보다 짧습니다.")
    if any(c in raw_pw for c in UNSAFE):
        print("[확인 필요] 비밀번호에 주소를 깨뜨릴 수 있는 문자가 있습니다.")
        ok = False
    if not ok:
        return False
    print("[OK] 주소 형식")

    with socket.socket() as s:
        s.settimeout(3)
        if s.connect_ex((u.hostname, port)) != 0:
            print(f"[대기] {port} 포트가 닫혀 있습니다. DB 컨테이너를 만들기 전이면 정상입니다.")
            return True
    print("[OK] 포트 열림")

    try:
        with psycopg.connect(dsn, connect_timeout=5) as c:
            print("[OK] 접속 성공, 비밀번호 일치")
            ver = c.execute("show server_version").fetchone()[0]
            has = c.execute("select 1 from pg_available_extensions where name = 'vector'").fetchone()
            print("PostgreSQL:", ver, "| pgvector:", "사용 가능" if has else "없음")
    except psycopg.OperationalError as e:
        msg = str(e).replace(pw, "****").replace(raw_pw, "****")
        if "password authentication failed" in msg:
            print("[확인 필요] 비밀번호 또는 사용자가 DB와 다릅니다.")
        else:
            print("[확인 필요] 접속 실패:", msg)
        return False
    return True


if __name__ == "__main__":
    check(dotenv_values(ROOT / ".env").get("RAG_DSN"))
