"""`.env` 점검: 비밀 값은 출력하지 않고 채워졌는지만 확인합니다.

    python -X utf8 check_env.py
"""
import re
from pathlib import Path

env = dict(re.findall(r"^([A-Z_]+)=(.*)$", Path(".env").read_text(encoding="utf-8"), re.M))
SECRETS = ["POSTGRES_PASSWORD", "RAG_PASSWORD", "VAULT_PASSWORD", "EMBED_API_KEY", "LLM_API_KEY", "HMAC_KEY"]
PLAIN = ["EMBED_PROVIDER", "EMBED_API_BASE", "EMBED_MODEL", "EMBED_DIM", "LLM_API_BASE", "LLM_MODEL"]


def status(v):
    v = v.strip()
    if not v:
        return "비어 있음"
    if v.startswith("change-me"):
        return "기본값 그대로(change-me)"
    return f"채워짐 ({len(v)}자)"


for k in SECRETS:
    print(f"{k:20}", status(env.get(k, "")))
print(f"{'HMAC 형식':20}", "OK" if re.fullmatch(r"[0-9a-f]{64}", env.get("HMAC_KEY", "").strip()) else "확인 필요")
dirty = [k for k in SECRETS if env.get(k, "") != env.get(k, "").strip().strip("\"'")]
print(f"{'앞뒤 공백·따옴표':20}", ", ".join(dirty) if dirty else "없음")
print(f"{'임베딩·LLM 키 같음':20}", env.get("EMBED_API_KEY") == env.get("LLM_API_KEY"))
for k in PLAIN:
    print(f"{k:20}", env.get(k))
