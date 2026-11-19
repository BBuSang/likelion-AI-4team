"""기획안 5-4 판단 기준 1~6 점검 (코랩에서 받은 벡터) → data/work/05_embed/audit.json, reports/5-4_임베딩.md

    python pipeline/embed_check_04.py                       # data/work/05_embed/kure_vectors.npz
    python pipeline/embed_check_04.py --npz 경로 --partial  # 일부만 만든 시험 파일
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np

from common import REPORTS, WORK

csv.field_size_limit(2**31 - 1)
E = WORK / "05_embed"
MAX_TOK = 512


def local_encoder():
    """로컬 CPU 질문 임베딩 (코랩 노트북 4번 셀과 같은 계산)"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")
    model = AutoModel.from_pretrained("nlpai-lab/KURE-v1").eval()

    @torch.inference_mode()
    def enc(texts, bs=8):
        out = []
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
            out.append(F.normalize(model(**x).last_hidden_state[:, 0].float(), dim=-1).numpy())
        return np.concatenate(out)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(E / "kure_vectors.npz"))
    ap.add_argument("--partial", action="store_true", help="앞 일부 조각만 들어 있는 시험 파일")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    ids, hashes, vecs = z["chunk_id"].tolist(), z["input_hash"].tolist(), z["vectors"]
    meta = json.loads(str(z["meta"]))
    rows = [json.loads(l) for l in open(WORK / "04_chunks" / "chunks_input.jsonl", encoding="utf-8")]
    if args.partial:
        rows = rows[:len(ids)]
    manifest = json.loads((E / "manifest.json").read_text(encoding="utf-8"))
    res = {}

    # 1. 개수·짝
    same_order = [r["chunk_id"] for r in rows] == ids
    res["1"] = {"벡터 수": len(ids), "조각 수(지금)": len(rows), "chunk_id 순서까지 같음": same_order,
                "pass": len(ids) == len(rows) and same_order}
    # 2. 입력 글 해시
    diff = [i for i, (r, h) in enumerate(zip(rows, hashes)) if r["input_hash"] != h]
    res["2"] = {"해시 다른 조각": len(diff), "묶음 지문(코랩)": meta["file_sha256"][:12],
                "묶음 지문(지금)": manifest["file_sha256"][:12], "pass": not diff and meta["file_sha256"] == manifest["file_sha256"]}
    # 3. 모양
    norms = np.linalg.norm(vecs, axis=1)
    res["3"] = {"차원": int(vecs.shape[1]), "NaN": int(np.isnan(vecs).sum()), "0벡터": int((norms < 1e-6).sum()),
                "길이 최소": round(float(norms.min()), 5), "길이 최대": round(float(norms.max()), 5),
                "pass": vecs.shape[1] == 1024 and not np.isnan(vecs).any() and abs(norms.min() - 1) <= 1e-3 and abs(norms.max() - 1) <= 1e-3}
    # 4. 반정밀 vs 정밀 (코랩에서 잰 값)
    s_idx, s32 = z["sample_idx"], z["sample_fp32"]
    fp = (vecs[s_idx] * s32).sum(1)
    res["4"] = {"정밀도": meta["dtype"], "장치": meta["device"], "조각": len(s_idx), "코사인 최소": round(float(fp.min()), 6),
                "pass": float(fp.min()) >= 0.999}
    # 5. 로컬 CPU 와 같은지 (조각 5개)
    enc = local_encoder()
    random.seed(5)
    pick = random.sample(range(len(rows)), min(5, len(rows)))
    t0 = time.time()
    local = enc([rows[i]["embed_text"] for i in pick])
    cpu_cos = (local * vecs[pick]).sum(1)
    res["5"] = {"조각": len(pick), "코사인 최소": round(float(cpu_cos.min()), 6), "코사인": [round(float(x), 6) for x in cpu_cos],
                "로컬 CPU 조각 1개 시간(초)": round((time.time() - t0) / len(pick), 2), "pass": float(cpu_cos.min()) >= 0.999}
    # 6. 맛보기 검색 (로컬 CPU 질문 벡터 → 코랩 조각 벡터)
    cites = {}
    for c in csv.DictReader(open(WORK / "04_chunks" / "chunks.csv", encoding="utf-8-sig")):
        cites[c["chunk_id"]] = (c["cite"], c["law_level"])
    qs = manifest["questions"]
    t0 = time.time()
    qv = enc(qs)
    q_time = (time.time() - t0) / len(qs)
    top = []
    for q, s in zip(qs, qv @ vecs.T):
        top.append({"질문": q, "1~5위": [(round(float(s[j]), 3), *cites.get(ids[j], ("?", "?"))) for j in np.argsort(-s)[:5]]})
    res["6"] = {"질문 1개 임베딩(초)": round(q_time, 2), "결과": top, "pass": True}
    res["meta"] = meta
    (E / "audit.json" if not args.partial else E / "audit_partial.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for k in "123456":
        print(k, "통과" if res[k]["pass"] else "실패", str({kk: vv for kk, vv in res[k].items() if kk not in ("결과", "pass")})[:200])
    if not args.partial:
        report(res)


def report(res):
    m = res["meta"]
    ok = lambda k: "✅ 통과" if res[k]["pass"] else "❌ 실패"
    L = []
    p = L.append
    p("# 5-4 임베딩 결과\n")
    p(f"모델 `{m['model']}` · {m['device']} · {m['dtype']} · {m['minutes']}분 · {m['made']} · 청크 규칙 `{m['chunk_rule']}` · "
      "코드 `colab/kure_embed.ipynb`(만드는 코드 `pipeline/make_colab_04.py`), `pipeline/embed_check_04.py`\n")
    p("## 판단 기준 점검 (기획안 5-4)\n")
    p("| # | 기준 | 결과 | 내용 |\n|---|---|---|---|")
    p(f"| 1 | 개수 | {ok('1')} | 벡터 {res['1']['벡터 수']:,} = 조각 {res['1']['조각 수(지금)']:,}, chunk_id 순서까지 같음 |")
    p(f"| 2 | 짝 확인 | {ok('2')} | 입력 글 해시가 다른 조각 {res['2']['해시 다른 조각']}, 묶음 지문 {res['2']['묶음 지문(코랩)']} = {res['2']['묶음 지문(지금)']} |")
    p(f"| 3 | 모양 | {ok('3')} | {res['3']['차원']}차원, NaN {res['3']['NaN']}, 0벡터 {res['3']['0벡터']}, 길이 {res['3']['길이 최소']}~{res['3']['길이 최대']} |")
    p(f"| 4 | 정밀도 | {ok('4')} | {res['4']['정밀도']} vs fp32 조각 {res['4']['조각']}개 코사인 최소 {res['4']['코사인 최소']} |")
    p(f"| 5 | 로컬 일치 | {ok('5')} | 코랩 벡터 vs 로컬 CPU 벡터 조각 {res['5']['조각']}개 코사인 최소 {res['5']['코사인 최소']} |")
    p(f"| 6 | 맛보기 | {ok('6')} | 예시 질문 {len(res['6']['결과'])}개 1~5위 (아래, 눈으로). 질문 1개 로컬 CPU {res['6']['질문 1개 임베딩(초)']}초 |\n")
    p("## 6. 맛보기 검색 (리트리버 평가 전, 조각 단위 1~5위)\n")
    for t in res["6"]["결과"]:
        p(f"**Q. {t['질문']}**\n")
        for s, cite, lv in t["1~5위"]:
            p(f"- {s:.3f} · {cite} ({lv})")
        p("")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "5-4_임베딩.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("reports/5-4_임베딩.md", len(L), "줄")


if __name__ == "__main__":
    main()
