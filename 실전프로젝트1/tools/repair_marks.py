"""생성 문서의 [[값|라벨]] 표시 보정 (hard 세트용).

어려운 표기(@ 대신 at, 공백 낀 주민번호, 호격이 붙은 이름 등)는 생성 모델이 문서에 넣고도 표시를 빠뜨리는 일이 많다
(시범 묶음에서 확인). 표시가 빠진 문서를 검수(qc_fresh.py)에서 버리기만 하면 어려운 문서만 골라서 빠지므로,
다른 호출(보정자)에게 표시된 문서를 주고 표시만 고친 문서 전체를 다시 받는다.

- 본문 글자는 바꾸지 못하게 한다: 표시를 지운 글자가 원래와 다르면 보정 결과의 표시만 순서대로 원래 본문에 옮겨 붙이고,
  옮길 수 없으면 원래 문서를 쓴다.
- 우리 탐지기(규칙·토큰)는 쓰지 않는다.
- 문서 내용은 출력하지 않는다. 바뀐 문서 수·표시 수만 남긴다.

사용법
  python -X utf8 tools/repair_marks.py --src raw_hard --dst raw_hard_fixed
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common 사용

from common import DATA, load_env

HEADER = re.compile(r"^=== *(\S+?) *\| *(.+?) *\| *(.+?) *$")
MARK = re.compile(r"\[\[([^\[\]|]+?)\|([a-z_]+)\]\]")

PROMPT = """아래는 개인정보를 [[값|라벨]]로 표시한 가상의 한국어 문서입니다. 표시가 빠진 개인정보에 표시를 추가하고,
잘못된 표시는 고쳐서 문서 전체를 다시 출력해 주세요. **표시 말고 본문 글자(띄어쓰기·문장부호 포함)는 한 글자도 바꾸지 마세요.**

라벨 7종: person(사람 이름, 외국인 포함), phone_number(개인 전화번호, 직원 직통·휴대폰·안심번호·인터넷전화 포함),
email(이메일, '@' 대신 at으로 쓴 것 포함), address(개인의 집·거주지·우편물 받는 주소),
resident_registration_number(주민등록번호·외국인등록번호), account_number(은행 계좌번호), card_number(카드번호)

표시 기준
- 값 자체만 [[ ]] 안에: 이름 뒤 조사·호격(아/야/이랑/이가)·호칭·직함·'드림'과 앞에 붙은 부서·호칭·은행·카드사 이름은 표시 밖
  예: [[민지|person]]야, 회원[[박정우|person]]님, 국민[[123-45-678901|account_number]], <[[abc at example.com|email]]>
- 같은 사람이 여러 번 나오면 나올 때마다, 영문 이름과 한글 이름을 함께 쓰면 각각 표시
- 표시하지 않는 것: 성과 직함만 있는 표현(김 과장), 회사 대표번호(1588-…), 내선번호, 회사·기관·은행·상품·가게 이름,
  사람이 아닌 뜻으로 쓴 단어(하늘, 나라 등), 회사·지점·매장·행사장·병원 같은 장소의 주소, 날짜, 금액, 사업자등록번호, 주문·접수·사건번호

설명 없이 고친 문서 본문만 출력하세요.

[문서]
{doc}"""


def strip(text):
    return MARK.sub(lambda m: m.group(1), text)


def transfer(fixed, body):
    """보정자가 본문을 조금 바꿨을 때: 보정 결과의 표시를 순서대로 원래 본문에서 찾아 다시 붙인다. 하나라도 못 찾으면 None."""
    plain, pos, spans = strip(body), 0, []
    for m in MARK.finditer(fixed):
        i = plain.find(m.group(1), pos)
        if i < 0:
            return None
        spans.append((i, i + len(m.group(1)), m.group(2)))
        pos = i + len(m.group(1))
    out, last = [], 0
    for s, e, lab in spans:
        out += [plain[last:s], f"[[{plain[s:e]}|{lab}]]"]
        last = e
    return "".join(out) + plain[last:]


def split_raw(path):
    """raw 파일 → [(헤더 줄, 본문)]"""
    docs, head, body = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if HEADER.match(line):
            if head:
                docs.append((head, "\n".join(body).strip()))
            head, body = line, []
        elif head:
            body.append(line)
    if head:
        docs.append((head, "\n".join(body).strip()))
    return docs


def repair(client, model, body, retries=3):
    for _ in range(retries):
        try:
            res = client.chat.completions.create(model=model, temperature=0,
                                                 messages=[{"role": "user", "content": PROMPT.format(doc=body)}])
            out = res.choices[0].message.content.strip()
            if out.startswith("```"):
                out = out.strip("`").split("\n", 1)[-1].strip()
            return out, res.usage.prompt_tokens, res.usage.completion_tokens
        except Exception:
            pass
    return None, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_hard")
    ap.add_argument("--dst", default="raw_hard_fixed")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI()

    work = DATA / "generated" / "_work"
    src, dst = work / a.src, work / a.dst
    dst.mkdir(parents=True, exist_ok=True)
    stats, tokens = Counter(), Counter()
    for path in sorted(src.glob("*.txt")):
        out_path = dst / path.name
        if out_path.exists() and out_path.stat().st_size > 0:
            continue  # 이어하기
        docs = split_raw(path)
        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(lambda d: repair(client, a.model, d[1]), docs))
        chunks = []
        for (head, body), (fixed, pt, ct) in zip(docs, results):
            tokens.update({"입력": pt, "출력": ct})
            if fixed is None:
                stats["보정 실패 → 원래 문서"] += 1
                chunks.append(f"{head}\n{body}")
            else:
                if strip(fixed) != strip(body):
                    moved = transfer(fixed, body)
                    if moved is None:
                        stats["본문이 바뀌고 표시를 옮기지 못함 → 원래 문서"] += 1
                        chunks.append(f"{head}\n{body}")
                        continue
                    stats["본문이 바뀌어 표시만 원래 본문으로 옮김"] += 1
                    fixed = moved
                before, after = len(MARK.findall(body)), len(MARK.findall(fixed))
                stats["보정 적용"] += 1
                stats["표시 수 (전)"] += before
                stats["표시 수 (후)"] += after
                stats["표시가 바뀐 문서"] += fixed != body
                chunks.append(f"{head}\n{fixed}")
        out_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(docs)}건", flush=True)
    report = {"보정 모델": a.model, **stats, "토큰": dict(tokens)}
    (work / f"repair_report_{a.src}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
