"""사람 잇기 비교: 2주차처럼 업로드 전체를 한꺼번에 묶기(batch) vs 최종 방식 한 건씩 저장하며 기존 인물과 잇기(one).

같은 데이터(2주차 holdout 등)를 두 방식으로 토큰화하고, 2주차 eval/check_token_consistency.py 로
미탐·쪼개짐·합쳐짐을 채점한다 (worklog 3-7-1 의 확인 방법). 한 건씩은 순서에 따라 결과가 달라질 수 있어 순서를 섞어 여러 번.

- 탐지는 2주차 탐지 캐시(outputs/detect_*.jsonl)를 씀 = 검수자가 탐지 결과를 그대로 승인했다고 가정
- 임베딩은 local (외부 API 안 씀). **로컬 개발 DB 의 rag·vault 데이터를 지우고 시작**하므로 --reset 필요
    python scripts/compare_linking.py --reset                     # holdout, 순서 3가지
    python scripts/compare_linking.py --reset --split synth2 --orders 2
"""
import argparse
import collections
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["EMBED_PROVIDER"] = "local"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W2 = ROOT.parents[1] / "실전프로젝트2"

from app.core import db  # noqa: E402
from app.ingest import service  # noqa: E402
from app.pii import person_resolve, pii_pipeline  # noqa: E402

SPLITS = {
    "holdout": ("data/holdout_labeled.jsonl", "data/holdout_persons_gold.jsonl"),
    "synth2": ("data/synth2/synth2_labeled.jsonl", "data/synth2/synth2_persons_gold.jsonl"),
    "synth3": ("data/synth3/synth3_labeled.jsonl", "data/synth3/synth3_persons_gold.jsonl"),
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score(gold, preds, tag, outdir):
    pred_path = outdir / f"pred_{tag}.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, str(W2 / "eval/check_token_consistency.py"), "--gold", str(gold),
                        "--pred", str(pred_path), "--out", str(outdir / f"cases_{tag}.csv")],
                       capture_output=True, text=True, encoding="utf-8", check=True,
                       env=os.environ | {"PYTHONIOENCODING": "utf-8"})
    lines = [l for l in r.stdout.splitlines() if "%" in l or "토큰" in l and "사례" not in l]
    return lines


def batch(docs, cache):
    spans, _ = person_resolve.resolve(docs, {d["id"]: cache[d["id"]] for d in docs})
    preds = []
    for d in docs:
        _, recs = pii_pipeline.pseudonymize(d["text"], spans[d["id"]])
        preds.append({"id": d["id"], "entities": [{k: r[k] for k in ("label", "start", "end", "token")} for r in recs]})
    return preds


def reset():
    with db.rag_conn() as c, c.cursor() as cur:
        cur.execute("TRUNCATE rag.chunks, rag.person_attrs, rag.persons")
    with db.vault_conn() as c, c.cursor() as cur:
        cur.execute("TRUNCATE vault.token_map, vault.person_mentions, vault.person_keys, vault.ingest_job CASCADE")


def one_by_one(docs, cache, seed):
    reset()
    order = docs[:]
    random.Random(seed).shuffle(order)
    by_text = {d["text"]: cache[d["id"]] for d in docs}
    service.pii_pipeline.detect = lambda text: by_text[text]  # 탐지 캐시 (모델 로딩 없음)
    content = "\n".join(json.dumps({"id": d["id"], "text": d["text"], "doc_type": d.get("doc_type")},
                                   ensure_ascii=False) for d in order)  # 정답 라벨은 안 넣음 → 전부 검수 대기
    job_id, parsed = service.create_job("compare@saebom.test", "jsonl", "compare", content)
    service.process_job(job_id, parsed)
    basis = collections.Counter()
    for row in service.list_docs(job_id):  # seq 순서 = 섞은 순서
        r = service.store(row["id"], actor="compare", action="approve")
        for g in r["links"]:
            basis[g["basis"]] += len(g["mentions"])
    with db.vault_conn() as c, c.cursor() as cur:
        cur.execute("SELECT ext_id, final_spans, text FROM vault.staging_doc WHERE job_id = %s", (job_id,))
        staged = cur.fetchall()
        cur.execute("SELECT doc_id, start_pos, token FROM vault.person_mentions")
        ptok = {(d, s): t for d, s, t in cur.fetchall()}
    preds = []
    for ext_id, spans, text in staged:
        doc_id = f"J{job_id}-{ext_id}"
        ents = []
        for s in spans:
            tok = ptok.get((doc_id, s["start"])) if s["label"] == "person" else pii_pipeline.make_token(
                s["label"], pii_pipeline.normalize(s["label"], text[s["start"]:s["end"]]) or text[s["start"]:s["end"]])
            ents.append({"label": s["label"], "start": s["start"], "end": s["end"], "token": tok})
        preds.append({"id": ext_id, "entities": ents})
    return preds, basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="holdout")
    ap.add_argument("--orders", type=int, default=3, help="한 건씩: 섞은 순서 몇 가지")
    ap.add_argument("--reset", action="store_true", help="로컬 DB 의 rag·vault 데이터를 지워도 됨")
    a = ap.parse_args()
    if not a.reset:
        sys.exit("로컬 DB 의 rag·vault 데이터를 지웁니다. 괜찮으면 --reset")
    data_path, gold_path = (W2 / p for p in SPLITS[a.split])
    docs = load(data_path)
    cache = {r["id"]: r["spans"] for r in load(W2 / f"outputs/detect_{data_path.stem}.jsonl")}
    outdir = Path(tempfile.mkdtemp(prefix="compare_linking_"))
    print(f"[{a.split}] 문서 {len(docs)}건, 결과 파일: {outdir}")

    print("\n== batch (2주차: 업로드 전체를 한꺼번에)")
    for l in score(gold_path, batch(docs, cache), "batch", outdir):
        print("  " + l)
    for seed in range(a.orders):
        preds, basis = one_by_one(docs, cache, seed)
        print(f"\n== one (한 건씩 + 기존 인물 잇기), 순서 seed={seed}")
        for l in score(gold_path, preds, f"one_{seed}", outdir):
            print("  " + l)
        print("  판단:", dict(basis.most_common()))
    reset()


if __name__ == "__main__":
    main()
