"""7-2 상품명 찾기(별칭) 확인 — 직원이 줄임말·별칭으로 부르는 새 질문으로 고치기 전후 비교

사용: python analysis/s18_alias_check.py gen      # data/eval/alias_check.jsonl (새 질문, 한 번만 만듦)
      python analysis/s18_alias_check.py score    # 고치기 전(analysis/archive/s12_retrieval_alias_old.py)·후 비교
- 새 질문: 상품공시 상품을 무작위 60개(seed 20261112, 같은 이름 중복 제외) 골라 CHAT_MODEL이 "영업점 직원이 그 상품을 줄여 부르는 말"을 넣은 질문 1개씩
  (정식 이름 그대로 쓰지 않게, 실제로 쓸 법한 줄임말·애칭). 검색 평가셋·test와 별개 → 7-1 test 실패를 보고 고친 규칙을 공정하게 확인
- 판정: 찾은 상품 이름에 정답 상품 이름이 있음(맞음) / 다른 상품만 찾음(틀림) / 못 찾음(좁히지 않음)
  · 틀림이 가장 나쁨 (엉뚱한 상품으로 좁혀 정답 문서가 빠짐), 못 찾음은 좁히지 않고 전체에서 검색하므로 덜 나쁨
- 이 질문으로는 규칙을 다시 고치지 않음 (고치면 또 새 질문이 필요)
"""
import csv
import importlib.util
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import load_env  # noqa: E402

OUT = ROOT / "data" / "eval" / "alias_check.jsonl"
OLD = ROOT / "analysis" / "archive" / "s12_retrieval_alias_old.py"   # 7-2 고치기 전 상품명 찾기 (세션 임시 폴더에서 옮김, 11-11 정리)
PROMPT = """너는 신한은행 영업점 직원이다. 주어진 상품을 동료에게 물을 때 실제로 부를 법한 **줄임말이나 애칭**으로 부르면서 질문 하나를 만든다.
- 정식 상품명을 그대로 쓰지 않는다 (예: '신한 장병내일준비 적금' → '장병적금' 또는 '장병내일준비', 'Hey Young(헤이영) 머니박스' → '헤이영 머니박스')
- 직원들이 흔히 쓰는 수준의 줄임말만 (아무도 못 알아들을 말은 안 됨)
- 질문 내용은 금리·가입대상·해지 등 아무거나 짧게
nickname에는 질문 속 상품 호칭만 적는다."""


class Q(BaseModel):
    nickname: str
    question: str


def gen():
    if OUT.exists():
        sys.exit("이미 만든 확인 질문이 있음 — 다시 만들지 않음")
    prods = list(csv.DictReader(open(ROOT / "data" / "meta" / "products.csv")))
    uniq = {}
    for p in prods:
        uniq.setdefault(p["name"], p)
    pick = random.Random(20261112).sample(sorted(uniq), 60)
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(name):
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=PROMPT, text_format=Q, input=f"상품: {name}")
        return {"name": name, "product_id": uniq[name]["product_id"], **r.output_parsed.model_dump()}

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, pick))
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{OUT.name}: {len(rows)}개")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # 세션 백업 파일은 프로젝트 밖에 있어 경로를 프로젝트 기준으로 맞춤
    m.ROOT, m.CH, m.EVAL, m.META = ROOT, ROOT / "data" / "chunks", ROOT / "data" / "eval", ROOT / "data" / "meta"
    return m


def score():
    rows = [json.loads(l) for l in open(OUT)]
    names = {p["product_id"]: p["name"] for p in csv.DictReader(open(ROOT / "data" / "meta" / "products.csv"))}
    for label, mod in (("고치기 전", load_module(OLD, "s12_old")), ("고친 뒤", load_module(ROOT / "analysis" / "s12_retrieval.py", "s12_new"))):
        idx = mod.Index()
        res = {"맞음": [], "틀림": [], "못 찾음": []}
        for r in rows:
            found = {names[p] for p in idx.detect_products(r["question"])}
            k = "못 찾음" if not found else "맞음" if r["name"] in found else "틀림"
            res[k].append((r, sorted(found)))
        print(f"[{label}] 맞음 {len(res['맞음'])} · 틀림 {len(res['틀림'])} · 못 찾음 {len(res['못 찾음'])} / {len(rows)}")
        for k in ("틀림", "못 찾음"):
            for r, f in res[k]:
                print(f"   {k}: '{r['nickname']}' (정답 {r['name']}) → {f[:3]}")


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
