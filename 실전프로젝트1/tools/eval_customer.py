"""고객 기준 채점 — "고객 개인정보를 얼마나 잘 잡느냐"를 성능 기준으로 봄 (팀 기준, 2026-10-23).

정답 중 고객(roles의 'C')의 개인정보만 대상으로 함: 고객 이름, 고객 전화·이메일·주소·주민번호·계좌·카드.
직원·외부인 이름을 잡은 것은 틀린 것으로 치지 않음 (더 가린 것일 뿐, 고객 정보가 새는 것은 아님).
run_compare.py와 정답 파일은 그대로 두고, 같은 함수로 계산함.

지표
    고객 R(partial) : 고객 개인정보 중 라벨이 같고 위치가 겹치게 잡은 비율
    고객 R(strict)  : 라벨·시작·끝이 정확히 같은 비율
    고객 mask_rec   : 고객 개인정보 글자 중 어떤 예측으로든 가려진 비율 (가장 중요)
    고객 미탐       : 고객 개인정보를 놓친 개수
    미끼 오탐       : 미끼 문서에서 정답(고객·직원 모두)과 겹치지 않는 예측 수

    고객 strict P   : 예측 중 고객 개인정보와 라벨·위치가 정확히 같은 비율.
                      직원·외부인 정답과 겹치는 예측은 빼고 계산 (맞은 것도 틀린 것도 아님)
    고객 일관성     : 같은 설정 3회 실행에서 결과가 같은 문서 비율 (직원 정답과 겹치는 예측은 빼고 비교)
    직원을 가리키는 자리 = 직원·외부인 정답 + 이름 없이 성+직함만 있는 말('정 부장', '김 대리님').
      이 자리를 잡은 예측은 정밀도·미끼 오탐·일관성 계산에서 모두 빼서 중립으로 둠 (팀 기준: 직원인 줄만 알면 됨)

사용법
    python tools/eval_customer.py test          # 모든 방식, 순위표
    python tools/eval_customer.py test --missed # 방식별 고객 미탐 목록도 출력
    python tools/eval_customer.py test --table  # 예전 비교표와 같은 열 (mask_rec, strict R/P/F1, 미끼 오탐, ms, 일관성)
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_compare as rc  # noqa: E402
from make_compare_table import SYSTEMS_NO_POST, decoy_fp  # noqa: E402

CUSTOMER_ROLES = {"C"}


def customer_gold(d):
    pids = {pid for r, pid in d["roles"].items() if r in CUSTOMER_ROLES}
    return [g for g in d["gold"] if g.get("person_id") in pids]


def score(docs, preds):
    n = strict = partial = chars = masked = 0
    missed = []
    for d, P in zip(docs, preds):
        for g in customer_gold(d):
            n += 1
            strict += any((p["label"], p["start"], p["end"]) == (g["label"], g["start"], g["end"]) for p in P)
            hit = any(p["label"] == g["label"] and rc.overlap(g, p) for p in P)
            partial += hit
            cov = set()
            for p in P:
                cov.update(range(max(g["start"], p["start"]), min(g["end"], p["end"])))
            chars += g["end"] - g["start"]
            masked += len(cov)
            if not hit:
                missed.append((d["id"], g["label"], g["text"], rc.ctx(d["text"], g["start"], g["end"])))
    return {"n": n, "strict_r": strict / n, "partial_r": partial / n, "mask_rec": masked / chars, "missed": missed}


# 이름 없이 성 + 직함으로 직원을 가리키는 말 ('정 부장', '김 대리님'). 정답에는 없지만 직원을 가리키므로 중립
SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부"
EMP_REF = re.compile(rf"(?<![가-힣])[{SURNAMES}]\s?(?:본부장|지점장|부장|과장|차장|대리|팀장|사원|주임|책임|선임|매니저|이사|실장|상무|전무)(?:님)?")


def employee_spans(d):
    """직원·외부인 정답 + 성+직함 표현 (고객 기준에서 잡아도 되고 안 잡아도 되는 자리)."""
    cust = customer_gold(d)
    spans = [g for g in d["gold"] if g not in cust]
    spans += [{"start": m.start(), "end": m.end()} for m in EMP_REF.finditer(d["text"])]
    return spans


def non_employee(d, P):
    """직원을 가리키는 자리(직원 정답, 성+직함)와 겹치는 예측을 뺀 나머지 (고객 기준에서 채점 대상인 예측)."""
    cust = customer_gold(d)
    emp = employee_spans(d)
    return [p for p in P if not (any(rc.overlap(g, p) for g in emp) and not any(rc.overlap(g, p) for g in cust))]


DECOY_TAGS = ("decoy_company_phone", "decoy_words", "decoy_numbers", "no_pii")


def customer_decoy_fp(docs, preds):
    """미끼 문서에서 고객 정답과도, 직원을 가리키는 자리와도 겹치지 않는 예측 수."""
    n = 0
    for d, P in zip(docs, preds):
        if any(t in d.get("hard_cases", []) for t in DECOY_TAGS):
            n += sum(1 for p in non_employee(d, P) if not any(rc.overlap(g, p) for g in customer_gold(d)))
    return n


def precision(docs, preds):
    tp = n = 0
    for d, P in zip(docs, preds):
        keys = {(g["label"], g["start"], g["end"]) for g in customer_gold(d)}
        Q = non_employee(d, P)
        n += len(Q)
        tp += sum((p["label"], p["start"], p["end"]) in keys for p in Q)
    return tp / n if n else 0.0


def table(split):
    from make_compare_table import load
    docs = rc.load_gold(f"data/pii_synth_finance_{split}.jsonl")
    rows = []
    for name, key, reruns in SYSTEMS_NO_POST:
        path = f"preds/{key}_{split}.jsonl"
        if not os.path.exists(path) or key in ("rules_v5", "rules_clean"):
            continue
        P, ms = load(path, docs)
        s = score(docs, P)
        p = precision(docs, P)
        f1 = 2 * p * s["strict_r"] / (p + s["strict_r"]) if p + s["strict_r"] else 0.0
        runs = [P] + [load(f"preds/{key}_{split}{r}.jsonl", docs)[0] for r in reruns
                      if os.path.exists(f"preds/{key}_{split}{r}.jsonl")]
        if len(runs) >= 3:
            filtered = [[non_employee(d, Pi) for d, Pi in zip(docs, run)] for run in runs]
            same = rc.consistency(filtered)[1]
        else:
            same = 1.0 if key.startswith("rules") else None
        rows.append((name, s, p, f1, customer_decoy_fp(docs, P), ms, same))
    rows.sort(key=lambda r: (-round(r[1]["mask_rec"], 3), -round(r[1]["strict_r"], 3), -round(r[2], 3),
                             r[4], -(r[6] or 0), r[5]))
    print(f"{split}: 고객 개인정보 {rows[0][1]['n']}개 기준 (후처리 없음, v5·clean 제외)\n")
    print("| 순위 | 방식 | mask_rec | strict R | strict P | strict F1 | 오탐(미끼) | ms/문서 | 일관성 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, (name, s, p, f1, fp, ms, same) in enumerate(rows, 1):
        cons = "-" if same is None else f"{same:.1%}"
        print(f"| {i} | {name} | {s['mask_rec']:.3f} | {s['strict_r']:.3f} | {p:.3f} | {f1:.3f} | {fp} | {ms:,.1f} | {cons} |")


def main():
    if "--table" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        return table(args[0] if args else "test")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = args[0] if args else "test"
    docs = rc.load_gold(f"data/pii_synth_finance_{split}.jsonl")
    rows = []
    for name, key, _ in SYSTEMS_NO_POST:
        path = f"preds/{key}_{split}.jsonl"
        if not os.path.exists(path) or key in ("rules_v5", "rules_clean"):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            P, ms = rc.load_pred(path, docs)
        s = score(docs, P)
        rows.append((name, s, decoy_fp(docs, P), ms))
    rows.sort(key=lambda r: (-round(r[1]["mask_rec"], 3), -round(r[1]["partial_r"], 3), r[2], r[3]))
    n = rows[0][1]["n"]
    print(f"{split}: 고객 개인정보 {n}개 기준 (후처리 없음, v5·clean 제외)\n")
    print("| 순위 | 방식 | 고객 mask_rec | 고객 R(partial) | 고객 R(strict) | 고객 미탐 | 미끼 오탐 | ms/문서 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, (name, s, fp, ms) in enumerate(rows, 1):
        print(f"| {i} | {name} | {s['mask_rec']:.3f} | {s['partial_r']:.3f} | {s['strict_r']:.3f} | "
              f"{len(s['missed'])} | {fp} | {ms:,.1f} |")
    if "--missed" in sys.argv:
        for name, s, _, _ in rows:
            if s["missed"]:
                print(f"\n== {name}: 고객 미탐 {len(s['missed'])}개")
                for m in s["missed"][:15]:
                    print("  ", *m)


if __name__ == "__main__":
    main()
