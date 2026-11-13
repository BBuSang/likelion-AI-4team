"""dev10/test10 각 50문항 생성 및 검수. 기존 방식의 35 회신 + 15 현장건의.
검색 성능으로 문항을 선별하지 않는다. 기존 세트의 정답 묶음 제외, 전체 원문 추가 검수,
형식/중복/유형/DB 연결 검사 후 저장. test 본문은 출력하지 않는다.
기존 파일은 덮어쓰지 않는다. 중단 시 별도 생성 캐시로 재개한다.
"""
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import d1_dev8_test8 as d8
import rag
from common import manifest

d7, gen, xf, EVAL = d8.d7, d8.gen, d8.xf, d8.EVAL
SEED = 20261112 + 10
d8.SKIP = ("dev10", "test10", "unanswerable_20")
d8.SEED = gen.SEED = d7.SEED = SEED
d8.N_EX = d7.N_EX = 3
d7.N_OLD = dict(d8.N_OLD)
gen.TYPES = {t: (d8.N_OLD[t], d8.N_OLD[t], *v[2:]) for t, v in gen.TYPES.items()}
gen.CACHE = EVAL / "_dev10_test10_raw.jsonl"
d7.EX_RAW = EVAL / "_dev10_test10_exmnt_raw.jsonl"
original_old, original_ex, original_call = gen.write_q, xf.write_q, gen.call_json


def call(system, user, effort="low"):
    for attempt in range(3):
        try:
            return original_call(system, user, effort)
        except Exception:
            if attempt == 2:
                raise RuntimeError("generation/check API failed; cached progress preserved") from None
            time.sleep(2 ** attempt)


gen.call_json = call


def full_review(item, producer):
    result = producer(item)
    if result.get("question") and result["check"]["answerable"] is True and result["check"]["leak"] is False:
        source = "\n".join(f"{k}: {item.get(k) or ''}" for k in ("_title", "_question", "_answer", "_reason"))
        check = call(gen.CHECK, f"[질문]\n{result['question']}\n\n[회신 전문]\n{source}", "medium")
        result["initial_check"] = result["check"]
        result["check"] = {k: check.get(k) for k in ("answerable", "leak", "reason")}
        result["full_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        result["gold_reference"] = {k[1:]: item.get(k) or "" for k in ("_title", "_question", "_answer", "_reason")}
        result["model"] = rag.CHAT_MODEL
    return result


gen.write_q = lambda item: full_review(item, original_old)
xf.write_q = lambda item: full_review(item, original_ex)


def norm(q):
    return re.sub(r"\W", "", q).lower()


def make_old_complete(rng):
    done = {x["source_case_key"]: x for x in map(json.loads, gen.CACHE.open(encoding="utf-8"))} if gen.CACHE.exists() else {}
    reserve, called = {}, 0
    for attempt in range(7):
        items, pool = gen.build(reserve)
        called += gen.generate(items, done)
        missing = {t: 2*n-gen.passed(items, done, t) for t, n in d7.N_OLD.items()}
        missing = {t: n for t, n in missing.items() if n > 0}
        if not missing:
            break
        if attempt < 6:
            for tag, count in missing.items():
                reserve[tag] = reserve.get(tag, 0) + 2*count
    dev, test, dropped = [], [], Counter()
    for tag, count in d7.N_OLD.items():
        passed, drop = d7.ok_rows(items, done, tag)
        dropped.update(drop)
        selected = passed[:2*count]
        rng.shuffle(selected)
        half = len(selected)//2
        dev.extend(selected[:half])
        test.extend(selected[half:])
    return dev, test, {"pool": pool, "called": called, "reserve": reserve, "dropped": dict(dropped), "short": missing}


def main():
    t0 = time.perf_counter()
    paths = [EVAL / f"{n}.jsonl" for n in ("dev10", "test10")]
    if any(p.exists() for p in paths):
        raise SystemExit("dev10/test10 already exists; refusing overwrite")
    used = d8.used_groups()
    previous_questions = set()
    previous_hashes = {}
    for p in EVAL.glob("*.jsonl"):
        if p.name.startswith("_") or p.stem in d8.SKIP:
            continue
        previous_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        for line in p.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("question"):
                previous_questions.add(norm(row["question"]))
    rng = random.Random(SEED)
    od, ot, stats_old = make_old_complete(rng)
    print(json.dumps({"phase": "replies", "dev": len(od), "test": len(ot), **stats_old}, ensure_ascii=False), flush=True)
    ed, et, stats_ex = d7.make_exmnt(rng)
    if stats_old["short"] or stats_ex["short"]:
        raise RuntimeError("insufficient verified items; cached progress preserved")
    for row in od + ot:
        row["corpus"] = "interp"
    for row in ed + et:
        row["corpus"] = "exmnt"
    sets = {"dev10": od + ed, "test10": ot + et}
    all_rows = sets["dev10"] + sets["test10"]
    assert len({r["gold_group_id"] for r in all_rows}) == 100
    assert not ({r["gold_group_id"] for r in all_rows} & used)
    assert len({norm(r["question"]) for r in all_rows}) == 100
    assert not ({norm(r["question"]) for r in all_rows} & previous_questions)
    with rag.connect() as conn:
        lookup = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases").fetchall()}
        lookup.update({r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.exmnt_cases").fetchall()})
    for name, rows in sets.items():
        assert len(rows) == 50
        assert Counter(r["corpus"] for r in rows) == {"interp": 35, "exmnt": 15}
        assert Counter(r["tag"] for r in rows if r["corpus"] == "interp") == d8.N_OLD
        assert Counter(r["tag"] for r in rows if r["corpus"] == "exmnt") == {t: 3 for t in xf.TYPES}
        rng.shuffle(rows)
        for i, row in enumerate(rows, 1):
            assert lookup[row["source_case_key"]] == row["gold_group_id"]
            assert row["check"]["answerable"] is True and row["check"]["leak"] is False
            assert row.get("gold_reference") and row["longest_copy"] <= gen.MAX_COPY
            lo, hi = gen.TYPES[row["tag"]][4] if row["corpus"] == "interp" else xf.LEN[row["tag"]]
            assert lo <= len(row["question"]) <= hi
            row.pop("rank", None)
            row.update(qid=f"{name.upper()}-{i:03d}", split="dev" if name == "dev10" else "test")
    # 부족하거나 검사를 통과하지 못한 세트는 최종 파일에 저장하지 않는다.
    for name, rows in sets.items():
        with (EVAL / f"{name}.jsonl").open("x", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    report = {"seed": SEED, "model": rag.CHAT_MODEL, "excluded_groups": len(used), "generation": {"replies": stats_old, "exmnt": stats_ex},
              "checks": {"unique_groups": 100, "previous_group_overlap": 0, "normalized_question_duplicates": 0,
                         "previous_exact_question_overlap": 0, "full_source_review_passed": 100, "db_link_passed": 100},
              "sets": {n: {"count": len(rs), "gubun": dict(Counter(r["gubun"] for r in rs)),
                             "tags": dict(Counter(r["corpus"] + ":" + r["tag"] for r in rs)),
                             "question_length": [min(len(r["question"]) for r in rs), max(len(r["question"]) for r in rs)],
                             "sha256": hashlib.sha256((EVAL / f"{n}.jsonl").read_bytes()).hexdigest()} for n, rs in sets.items()},
              "previous_set_sha256": previous_hashes, "seconds": round(time.perf_counter() - t0, 1),
              "limitations": ["Synthetic single-gold data, not human gold", "Same model for generation and checking", "Semantic overlap beyond group IDs not ruled out"]}
    (EVAL / "d1_dev10_test10_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest("d1_dev10_test10", **report)
    print(json.dumps({k: v for k, v in report.items() if k != "previous_set_sha256"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
