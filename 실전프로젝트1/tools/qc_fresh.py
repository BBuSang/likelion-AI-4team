"""2차 평가 세트(synth_fresh) 정답 검수: 생성 모델이 표시를 빠뜨리거나 잘못 표시한 문서를 평가에서 뺀다.

생성 정답은 모델이 직접 [[값|라벨]]로 표시해서, 표시 누락(개인 주소를 그냥 둠 등)이 있으면
탐지기가 맞게 잡아도 오탐으로 채점된다. 그래서 다른 호출(검수자)에게 표시된 문서를 보여 주고
"표시가 빠진 개인정보"와 "개인정보가 아닌데 표시한 것"을 찾게 한 뒤, 하나라도 나온 문서는 제외한다.

편향 주의
- 검수자는 우리 탐지기(규칙·토큰)를 쓰지 않는다. 우리 탐지기 결과로 정답을 고치면 채점이 탐지기 쪽으로 기운다.
- 검수자가 문제를 찾은 문서를 "고치지 않고 버린다": 검수자의 판단을 정답에 섞지 않기 위해.
- 문서 내용은 출력하지 않는다(평가 세트를 규칙 작성자가 보지 않도록). 제외 사유별 개수만 남긴다.

사용법
  python -X utf8 tools/qc_fresh.py              # synth_fresh_all.jsonl → synth_fresh.jsonl + qc_fresh_report.json
  python -X utf8 tools/qc_fresh.py --set hard   # synth_hard_all.jsonl → synth_hard.jsonl (표시 보정 repair_marks.py 뒤)
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common 사용

from common import DATA, LABELS, load_env

PROMPT = """아래는 개인정보를 [[값|라벨]]로 표시한 가상의 한국어 문서입니다. 표시가 정확한지 검수해 주세요.

라벨 7종: person(사람 이름, 외국인 포함), phone_number(개인 전화번호, 직원 직통·휴대폰 포함), email,
address(개인의 집·거주지·우편물 받는 주소), resident_registration_number(주민등록번호·외국인등록번호),
account_number(은행 계좌번호, 은행 이름 제외), card_number(카드번호)

개인정보가 아닌 것(표시하지 않는 것이 맞음): 성과 직함만 있는 표현(김 과장), 회사 대표번호(1588-…), 내선번호,
회사·기관·은행·상품·가게 이름, 회사·지점·매장·행사장·병원 같은 장소의 주소, 날짜, 금액, 사업자등록번호, 주문·접수·사건번호,
문서에 '계좌 아님'·'대표번호'처럼 개인정보가 아니라고 적힌 값

팀 표시 기준 (아래는 모두 개인정보로 표시하는 것이 맞음)
- 성 없이 이름만 쓴 경우, 외자·띄어 쓴 이름, 일반 단어와 같은 이름(하늘, 보람 등)도 사람을 가리키면 person
  (값에는 이름만: 호격 아/야, 조사, 님·씨, 직함, 앞에 붙은 부서·호칭은 표시 밖)
- 지역번호(02, 031 …)·070·050 번호도 개인이나 직원의 연락처로 쓰였으면 phone_number (회사 대표번호로 쓰인 것만 아님)
- '@' 대신 at, [at]으로 쓴 이메일도 email
- 휴대폰 번호 모양의 계좌번호도 계좌로 쓰였으면 account_number (은행·카드사 이름은 표시 밖)
- 같은 사람을 여러 번, 다른 표기로 써도 매번 표시
위 기준에 맞는 표시는 wrong이 아닙니다. 기준에 비추어 확실히 틀린 것만 적어 주세요.

찾을 것
1. missing: 위 7종에 해당하는데 [[ ]] 표시가 없는 값 (문서에 적힌 글자 그대로)
2. wrong: 표시되어 있지만 개인정보가 아니거나 라벨이 틀린 것

JSON만 출력: {{"missing": [{{"value": "...", "label": "..."}}], "wrong": [{{"value": "...", "label": "...", "reason": "..."}}]}}
문제가 없으면 {{"missing": [], "wrong": []}}

[문서]
{doc}"""


def marked(doc):
    """정답 위치로 [[값|라벨]] 표시를 다시 붙인다."""
    out, last = [], 0
    for e in sorted(doc["entities"], key=lambda e: e["start"]):
        out.append(doc["text"][last:e["start"]])
        out.append(f"[[{doc['text'][e['start']:e['end']]}|{e['label']}]]")
        last = e["end"]
    out.append(doc["text"][last:])
    return "".join(out)


def review(client, model, doc, retries=3):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT.format(doc=marked(doc))}])
            r = json.loads(res.choices[0].message.content)
            return r, res.usage.prompt_tokens, res.usage.completion_tokens
        except Exception:
            pass
    return None, 0, 0


def problems(doc, r):
    """검수 결과 중 실제로 확인되는 문제만 센다: missing 값이 원문에 있고 기존 정답과 겹치지 않을 때."""
    if r is None:
        return ["검수 실패"]
    out = []
    spans = [(e["start"], e["end"]) for e in doc["entities"]]
    for m in r.get("missing", []):
        v, lab = str(m.get("value", "")).strip(), m.get("label")
        if not v or lab not in LABELS:
            continue
        pos = doc["text"].find(v)
        while pos != -1:
            if not any(s < pos + len(v) and pos < t for s, t in spans):
                out.append(f"누락:{lab}")
                break
            pos = doc["text"].find(v, pos + 1)
    for w in r.get("wrong", []):
        if w.get("value"):
            out.append(f"잘못 표시:{w.get('label')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 몇 건만 (시험용)")
    ap.add_argument("--set", default="fresh", choices=["fresh", "hard"])
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI()

    work = DATA / "generated" / "_work"
    src = work / f"synth_{a.set}_all.jsonl"
    docs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    if a.limit:
        docs = docs[:a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda d: review(client, a.model, d), docs))

    keep, reasons, docs_by_reason = [], Counter(), Counter()
    for d, (r, _, _) in zip(docs, results):
        ps = problems(d, r)
        reasons.update(ps)
        docs_by_reason.update(set(p.split(":")[0] for p in ps))
        if not ps:
            d["split"] = f"synth_{a.set}"
            keep.append(d)
    # fresh 는 바로 채점용(generated/), hard 는 split_hard.py 로 반을 나누기 전 중간 파일(_work/)
    out = (DATA / "generated" if a.set == "fresh" else work) / f"synth_{a.set}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in keep:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    labels = Counter(e["label"] for d in keep for e in d["entities"])
    report = {
        "검수 모델": a.model, "검수 문서": len(docs), "남긴 문서": len(keep), "뺀 문서": len(docs) - len(keep),
        "뺀 사유별 문서 수": dict(docs_by_reason), "찾은 문제 (라벨별)": dict(reasons.most_common()),
        "남긴 문서 정답 수": sum(labels.values()), "남긴 문서 라벨별": dict(labels.most_common()),
        "남긴 문서 중 개인정보 없는 문서": sum(not d["entities"] for d in keep),
        "토큰": {"입력": sum(r[1] for r in results), "출력": sum(r[2] for r in results)},
    }
    (work / f"qc_{a.set}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
