"""데이터 로드, 예측 저장, .env 읽기 등 탐지기 공통 기능."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).parent          # 실전프로젝트1 (preds/, results/ 기준)
DATA = ROOT / "data"                  # 정답 데이터 (제공 데이터, 엣지 케이스, 생성 데이터, 라벨별 CSV)
LABELS = ["person", "phone_number", "email", "address",
          "resident_registration_number", "account_number", "card_number"]


def load_docs(split: str) -> list:
    # edge: make_edge_cases.py 로 만든 엣지 케이스 시험 문서, synth_*: generate_synthetic.py 로 만든 대량 가상 문서
    if split == "edge":
        path = DATA / "edge_cases.jsonl"
    elif split == "holdout":  # 멘토 배포 holdout (받은 그대로 holdout/ 폴더에 둠)
        path = ROOT / "holdout" / "holdout_labeled.jsonl"
    elif split.startswith("synth"):
        path = DATA / "generated" / f"{split}.jsonl"
    else:
        path = DATA / f"pii_synth_finance_{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_env():
    """.env 를 이 폴더 → 상위 폴더 순서로 찾아 환경변수로 등록.
    키 이름은 'openai-api-key' 처럼 하이픈을 써도 OPENAI_API_KEY 로 인식한다."""
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip().upper().replace("-", "_"), v.strip().strip('"\''))


def dedupe(text: str, ents: list) -> list:
    """앞뒤 공백 제거 + 중복 제거. 채점기는 중복 예측을 오탐으로 세기 때문에 반드시 거친다."""
    seen, out = set(), []
    for e in ents:
        s, t = e["start"], e["end"]
        while s < t and text[s].isspace():
            s += 1
        while t > s and text[t - 1].isspace():
            t -= 1
        k = (e["label"], s, t)
        if s < t and e["label"] in LABELS and k not in seen:
            seen.add(k)
            out.append({**e, "start": s, "end": t, "text": text[s:t]})
    return sorted(out, key=lambda e: (e["start"], e["end"]))


def find_all(text: str, value: str, ignore_space: bool = False) -> list:
    """value 가 원문에 나오는 모든 (start, end).
    ignore_space=True(이름용)면 글자 사이 공백 유무를 무시하고 찾는다
    (LLM 이 '윤 선영'만 답했을 때 '윤선영'도, 그 반대도 찾기 위해).
    그 외에는 정확히 일치하는 곳이 없을 때만 공백을 무시하고 다시 찾는다."""
    chars = [re.escape(c) for c in value if not c.isspace()]
    if ignore_space and chars:
        # 앞 글자가 한글이면 다른 이름의 일부다: '소율'이 '이소율' 안에서 잡히지 않도록
        pat = r"(?<![가-힣])" + r" ?".join(chars)
        return [(m.start(), m.end()) for m in re.finditer(pat, text)]
    hits = [(m.start(), m.end()) for m in re.finditer(re.escape(value), text)]
    if not hits and chars:
        hits = [(m.start(), m.end()) for m in re.finditer(r"\s?".join(chars), text)]
    return hits


def save_preds(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
