"""DB 컨테이너 실행: .env의 RAG_DSN에서 접속 값을 읽어 docker compose에 넘기고 접속을 확인한다.
값은 환경 변수로만 전달하며 파일에 쓰거나 출력하지 않는다.
사용: python db_up.py
"""
import contextlib
import io
import os
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import psycopg
from dotenv import dotenv_values

from check_db_env import check
from common import ROOT

COMPOSE = ROOT / "db" / "docker-compose.yml"


def quiet_check(dsn):
    """점검 결과에서 판정 줄만 보여준다. 계정 이름 등 값이 담긴 줄은 내보내지 않는다."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = check(dsn)
    for line in buf.getvalue().splitlines():
        if line.startswith("["):
            print(line)
    return ok


def main():
    dsn = dotenv_values(ROOT / ".env").get("RAG_DSN")
    if not quiet_check(dsn):
        sys.exit("RAG_DSN을 먼저 고쳐 주세요.")
    u = urlsplit(dsn)
    if u.hostname not in ("localhost", "127.0.0.1"):
        sys.exit("RAG_DSN의 호스트가 내 컴퓨터가 아니어서 컨테이너를 만들지 않습니다.")
    pw = unquote(u.password)
    env = {**os.environ, "RAG_DB_USER": unquote(u.username), "RAG_DB_PASSWORD": pw, "RAG_DB_NAME": u.path.lstrip("/"), "RAG_DB_PORT": str(u.port)}
    r = subprocess.run(["docker", "compose", "-p", "interp_rag", "-f", str(COMPOSE), "up", "-d", "--wait"], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit("컨테이너 실행 실패: " + (r.stderr or r.stdout).replace(pw, "****")[-800:])
    print("[OK] 컨테이너 실행")
    if not quiet_check(dsn):
        sys.exit("컨테이너는 떴지만 접속에 실패했습니다.")
    with psycopg.connect(dsn, connect_timeout=5) as c:
        ext = [x[0] for x in c.execute("select extname from pg_extension order by 1").fetchall()]
        ver = c.execute("show server_version").fetchone()[0]
    print("PostgreSQL", ver, "| 설치된 확장", ext)


if __name__ == "__main__":
    main()
