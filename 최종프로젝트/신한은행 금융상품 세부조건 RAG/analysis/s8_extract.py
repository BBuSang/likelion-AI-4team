"""3단계 LLM 추출 준비 — 입력 만들기, 결과 검증, 계약(프롬프트·스키마·DB) 일치 검사

사용:
  python analysis/s8_extract.py check            # models.py ↔ extract_prompt.md ↔ schema.sql 일치 검사 (API 호출 없음)
  python analysis/s8_extract.py input D156       # 문서 1건의 LLM 입력을 화면에 출력 (API 호출 없음)
  python analysis/s8_extract.py targets          # 추출 대상 문서 목록과 입력 크기
  python analysis/s8_extract.py run D156 …       # 시험 추출 → data/extract/trial
  python analysis/s8_extract.py run-all          # 대상 130개 전체 → data/extract/full (이어하기·병렬 6·재시도 3)

입력: 정형 문구가 아닌 섹션의 정리 본문(2-6 text)을 `[S{seq}] 섹션 경로 (p.쪽)` 머리와 함께.
검증: 각 금리·수수료·혜택의 source_text가 section_seq 섹션 text 안에 있는지 (공백 무시) + 숫자가 source_text에 있는지.
"""
import csv
import json
import re
import sys
import typing
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from models import Benefit, DocumentExtraction, Fee, Rate  # noqa: E402

SECS, META = ROOT / "data" / "sections", ROOT / "data" / "meta"
TARGET_TYPES = {"상품설명서", "간이투자설명서", "투자설명서"}   # 상품 조건 표가 있는 문서 (특약·약관·약정서는 평가 후 결정)
DETAIL_KIND = {"저축상품": "deposit", "대출상품": "loan", "기타상품": "invest", "복합금융상품": "invest"}


def ns(t: str) -> str:
    return re.sub(r"\s", "", t)


def doc_info():
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    prods = {r["product_id"]: r for r in csv.DictReader(open(META / "products.csv"))}
    cats = defaultdict(set)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        cats[r["doc_id"]].add(prods[r["product_id"]]["category"])
    return docs, cats


def load_sections(doc_id):
    return [json.loads(l) for l in open(SECS / f"{doc_id}.jsonl")]


def build_input(doc_id, docs=None, cats=None) -> str:
    if docs is None:
        docs, cats = doc_info()
    d, c = docs[doc_id], sorted(cats[doc_id])
    kinds = {DETAIL_KIND.get(x, "none") for x in c}
    kind = kinds.pop() if len(kinds) == 1 else "none"
    head = f"문서: {d['title']} ({d['doc_type']}) / 연결 상품 분류: {', '.join(c)} / detail_kind: {kind}"
    body = [f"[S{s['seq']}] {s['section_path']} (p.{s['page_start']})\n{s['text']}"
            for s in load_sections(doc_id) if not s["is_boilerplate"]]
    return head + "\n\n" + "\n\n".join(body)


def lines_in_order(src: str, sec: str) -> bool:
    """source_text의 각 줄(공백 무시)이 섹션 안에 원문 순서대로 있는지 — 표에서 줄 단위로 발췌한 근거도 허용"""
    pos = 0
    for line in (ns(l) for l in src.split("\n")):
        if not line:
            continue
        i = sec.find(line, pos)
        if i < 0:
            return False
        pos = i + len(line)
    return True


def has_number(v: float, text: str) -> bool:
    """text 안의 숫자 중 v와 같은 값이 있는지 (0.01 = 0.01000, 1,000 = 1000, 1만원 = 10000, 3만 5천원 = 35000, 1천만원, 무이자 = 0)"""
    cands = set()
    for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            cands.add(float(x.replace(",", "")))
        except ValueError:
            pass
    unit = {"천만": 1e7, "백만": 1e6, "억": 1e8, "만": 1e4, "천": 1e3, "백": 1e2}
    for m in re.finditer(r"((?:\d[\d,]*(?:\.\d+)?\s*(?:천만|백만|[억만천백])\s*)+)", text):
        total = 0.0
        for num, u in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(천만|백만|[억만천백])", m.group(1)):
            total += float(num.replace(",", "")) * unit[u]
        cands.add(total)
    if re.search(r"무이자|없음|면제", text):
        cands.add(0.0)
    return any(abs(c - abs(v)) < 1e-9 for c in cands)


def verify(doc_id, ext: DocumentExtraction):
    """근거 검사 → [(종류, 순번, 문제)]
    1) section_seq 섹션이 있는지 2) source_text의 각 줄이 그 섹션에 원문 순서대로 있는지 3) 금리·정률·혜택 숫자가 source_text에 있는지"""
    secs = {s["seq"]: ns(s["text"]) for s in load_sections(doc_id)}
    problems = []
    for kind, items, nums in (("rate", ext.rates, ("rate_min", "rate_max")),
                              ("fee", ext.fees, ("rate_pct",)),
                              ("benefit", ext.benefits, ("value_num",))):
        for i, it in enumerate(items):
            sec = secs.get(it.section_seq)
            if sec is None:
                problems.append((kind, i, f"없는 섹션 S{it.section_seq}"))
            elif not lines_in_order(it.source_text, sec):
                problems.append((kind, i, f"source_text가 S{it.section_seq}에 없음"))
            for f in nums:
                v = getattr(it, f)
                if v is not None and not has_number(v, it.source_text):
                    problems.append((kind, i, f"{f}={v:g}가 source_text에 없음"))
    return problems


def check_contract():
    """models.py ↔ extract_prompt.md ↔ schema.sql"""
    prompt = (ROOT / "models" / "extract_prompt.md").read_text()
    sql = (ROOT / "models" / "schema.sql").read_text()
    errs = []

    def cols(table):
        m = re.search(rf"CREATE TABLE {table} \((.*?)\n\);", sql, re.S)
        return {l.strip().split()[0] for l in m.group(1).split("\n") if l.strip() and not l.strip().startswith("--")}

    for model, table in ((Rate, "doc_rate"), (Fee, "doc_fee"), (Benefit, "doc_benefit")):
        missing = set(model.model_fields) - cols(table)
        if missing:
            errs.append(f"{table}에 없는 필드: {sorted(missing)}")
        kind_field = next(f for f in model.model_fields if f.endswith("_kind"))
        for v in typing.get_args(model.model_fields[kind_field].annotation):
            if v not in sql:
                errs.append(f"schema.sql 주석에 없는 {kind_field} 값: {v}")
    doc_cols = cols("document")
    for f in DocumentExtraction.model_fields:
        if f in ("rates", "fees", "benefits", "doc_title_in_text"):
            continue
        col = "extract_notes" if f == "notes" else f
        if col not in doc_cols:
            errs.append(f"document에 없는 필드: {f}")
    for v in typing.get_args(Rate.model_fields["rate_kind"].annotation):
        if v not in prompt and v not in ("base", "preferential", "max", "early_termination", "after_maturity",
                                          "loan_base", "loan_spread", "loan_preferential", "loan_final"):
            errs.append(f"프롬프트에 설명 없는 rate_kind: {v}")
    if "DocumentExtraction" not in prompt or "ProductExtraction" in prompt or "sections:" in prompt:
        errs.append("프롬프트가 옛 스키마를 가리킴")
    # OpenAI 구조화 출력(strict)으로 바꿀 수 있는지
    from openai.lib._pydantic import to_strict_json_schema
    try:
        to_strict_json_schema(DocumentExtraction)
    except Exception as e:  # noqa: BLE001
        errs.append(f"strict JSON schema 변환 실패: {e}")
    return errs


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


def extract_one(client, model, prompt, d, docs, cats, out_dir: Path, tries=3):
    import time
    for t in range(tries):
        try:
            resp = client.responses.parse(model=model, instructions=prompt,
                                          input=build_input(d, docs, cats), text_format=DocumentExtraction)
            ext = resp.output_parsed
            if ext is None:
                raise RuntimeError("구조화 출력 없음")
            break
        except Exception as e:  # noqa: BLE001 — 일시 오류는 다시 시도, 끝내 실패하면 기록
            if t == tries - 1:
                (out_dir / f"{d}.error.txt").write_text(f"{type(e).__name__}: {e}")
                return f"{d}: 실패 {type(e).__name__}"
            time.sleep(5 * (t + 1))
    probs = verify(d, ext)
    u = resp.usage
    status = "needs_review" if (probs or ext.notes) else "ok"
    rec = {"doc_id": d, "model": model, "status": status, "usage": {"input": u.input_tokens, "output": u.output_tokens},
           "problems": probs, "extraction": ext.model_dump()}
    (out_dir / f"{d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    (out_dir / f"{d}.error.txt").unlink(missing_ok=True)
    n = len(ext.rates) + len(ext.fees) + len(ext.benefits)
    return (f"{d}: 금리 {len(ext.rates)} 수수료 {len(ext.fees)} 혜택 {len(ext.benefits)} | 근거 문제 {len(probs)}/{n} "
            f"| notes {len(ext.notes)} | {status} | 토큰 {u.input_tokens:,}/{u.output_tokens:,}")


def run(doc_ids, out_dir: Path, workers=1, skip_done=False):
    """CHAT_MODEL로 구조화 출력 추출 → out_dir/{doc_id}.json (결과·상태·토큰·근거 검사).
    status: 근거 검사 문제나 notes가 있으면 needs_review, 아니면 ok. skip_done이면 이미 끝난 문서는 건너뜀(이어하기)"""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None, timeout=600)
    prompt = (ROOT / "models" / "extract_prompt.md").read_text()
    docs, cats = doc_info()
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in doc_ids if not (skip_done and (out_dir / f"{d}.json").exists())]
    print(f"대상 {len(doc_ids)}개 중 {len(todo)}개 실행 (병렬 {workers})", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for i, line in enumerate(ex.map(lambda d: extract_one(client, env["CHAT_MODEL"], prompt, d, docs, cats, out_dir), todo), 1):
            print(f"[{i}/{len(todo)}] {line}", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "run":
        run(sys.argv[2:], ROOT / "data" / "extract" / "trial")
        return
    if cmd == "run-all":   # 추출 대상 130개 전체, 이어하기, 병렬 6
        docs, _ = doc_info()
        targets = sorted(d for d, r in docs.items() if r["doc_type"] in TARGET_TYPES)
        run(targets, ROOT / "data" / "extract" / "full", workers=6, skip_done=True)
        return
    if cmd == "check":
        errs = check_contract()
        print("계약 검사:", "통과" if not errs else "")
        for e in errs:
            print(" -", e)
    elif cmd == "input":
        print(build_input(sys.argv[2]))
    elif cmd == "targets":
        docs, cats = doc_info()
        rows = [(d, r["doc_type"], len(build_input(d, docs, cats))) for d, r in docs.items() if r["doc_type"] in TARGET_TYPES]
        L = sorted(n for *_, n in rows)
        print(f"추출 대상 {len(rows)}개 문서, 입력 글자 합 {sum(L):,} (중앙 {L[len(L)//2]:,} / 최대 {L[-1]:,})")


if __name__ == "__main__":
    main()
