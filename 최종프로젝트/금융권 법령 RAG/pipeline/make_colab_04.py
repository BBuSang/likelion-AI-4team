"""기획안 5-4 코랩 노트북 만들기 → colab/kure_embed.ipynb

노트북은 코랩(GPU)과 로컬(CPU 시험) 둘 다에서 돈다.
  - 코랩: kure_input.zip 올리기 → 전부 실행 → kure_vectors.npz 내려받기
  - 로컬 시험: LOCAL_TEST_N=30 KURE_ZIP=… OUT_NPZ=… 로 앞 30조각만

임베딩 방식은 로컬 질문 임베딩과 똑같이 transformers 로 직접 (CLS 토큰 → 길이 1로 정규화, 최대 512토큰).
KURE-v1 의 sentence-transformers 설정(풀링 CLS + Normalize)과 같은 계산이다.

    python pipeline/make_colab_04.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "kure_embed.ipynb"

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
    MD("""
# 법령 RAG 5-4 임베딩 (KURE-v1, 코랩 GPU)

기획안 5-4 단계. 5-3에서 만든 조각 전부(개수·지문은 2번 셀이 보여 줌)를 `nlpai-lab/KURE-v1`로 벡터(1024칸)로 바꿉니다.

## 쓰는 법
1. 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** (또는 다른 GPU) → 저장
2. 메뉴 **런타임 → 모두 실행**
3. 2번 셀에서 **파일 선택** 버튼이 나오면 `kure_input.zip` 을 고릅니다 (로컬 `최종프로젝트/금융권 법령 RAG/data/work/05_embed/kure_input.zip`, 8MB)
4. 끝나면 `kure_vectors.npz` 가 자동으로 내려받아집니다 (약 130MB, 지금 T4 기준 약 5분) → `최종프로젝트/금융권 법령 RAG/data/work/05_embed/` 에 넣고 로컬에서
   `python pipeline/embed_check_04.py` 를 실행합니다

## 이 노트북이 스스로 확인하는 것 (기획안 5-4 판단 기준)
| # | 기준 | 어디서 |
|---|---|---|
| 1 | 벡터 수 = 조각 수, chunk_id 짝 | 2번·7번 셀 |
| 2 | 입력 글 해시(`input_hash`) = 조각 글 | 2번 셀 (하나라도 다르면 멈춤) |
| 3 | 1024칸, NaN·0벡터 0, 길이 1 ± 0.001 | 7번 셀 |
| 4 | GPU 반정밀(fp16) vs 정밀(fp32) 코사인 ≥ 0.999 (조각 20개) | 5번·7번 셀 |
| 5 | 로컬 CPU 벡터와 같은지 | 로컬 `embed_check_04.py` |
| 6 | 예시 질문으로 검색해 1~5위가 말이 되는지 | 8번 셀 (눈으로) |

> 계산 방식: 토크나이저로 최대 512토큰 → 모델 → 첫 토큰(CLS) 벡터 → 길이 1로 정규화.
> 로컬 질문 임베딩도 같은 코드라서 조각 벡터와 질문 벡터가 같은 공간에 있습니다.
"""),
    CODE("""
# 1. 준비: 장치 확인
import os, sys, json, time, hashlib, zipfile, random
import numpy as np, torch, torch.nn.functional as F

IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = int(os.environ.get("LOCAL_TEST_N", "0"))   # 로컬 시험용: 앞 N조각만 (코랩에서는 0 = 전부)
MODEL, MAX_TOK = "nlpai-lab/KURE-v1", 512
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("코랩" if IN_COLAB else "로컬", "| 장치", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "",
      "| torch", torch.__version__)
if IN_COLAB:
    assert dev == "cuda", "GPU가 없습니다: 런타임 → 런타임 유형 변경 → GPU 를 고르고 다시 실행하세요"
"""),
    CODE("""
# 2. 입력 올리기 + 확인 (파일 지문, 조각 수, 조각마다 입력 글 해시, chunk_id 중복)
if IN_COLAB:
    from google.colab import files
    print("kure_input.zip 을 고르세요")
    up = files.upload()
    ZIP = next(k for k in up if k.endswith(".zip"))
else:
    ZIP = os.environ["KURE_ZIP"]
with zipfile.ZipFile(ZIP) as z:
    raw = z.read("chunks_input.jsonl")
    manifest = json.loads(z.read("manifest.json"))
assert hashlib.sha256(raw).hexdigest() == manifest["file_sha256"], "파일 지문이 다릅니다 (다른 묶음?)"
rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
assert len(rows) == manifest["n_chunks"], "조각 수가 manifest 와 다릅니다"
bad = [r["chunk_id"] for r in rows if hashlib.sha256(r["embed_text"].encode("utf-8")).hexdigest() != r["input_hash"]]
assert not bad, f"입력 글 해시가 다른 조각 {len(bad)}개: {bad[:3]}"
assert len({r["chunk_id"] for r in rows}) == len(rows), "chunk_id 중복"
if LOCAL_TEST:
    rows = rows[:LOCAL_TEST]
print(f"조각 {len(rows):,}개 · 지문 {manifest['file_sha256'][:12]} · 청크 규칙 {manifest['chunk_rule']} · 묶은 날 {manifest['made_on']}")
"""),
    CODE("""
# 3. 모델 불러오기 + 토큰 길이 확인 (512토큰을 넘으면 모델이 뒤를 잘라 버리므로 0이어야 함)
from transformers import AutoTokenizer, AutoModel
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()
print(f"모델 로드 {time.time() - t0:.1f}초 · 최대 길이 설정 {tok.model_max_length}")
lens = [len(x) for x in tok([r["embed_text"] for r in rows], add_special_tokens=True)["input_ids"]]
print(f"토큰 수: 평균 {np.mean(lens):.0f} · 최대 {max(lens)} · 512 초과 {sum(l > MAX_TOK for l in lens)}")
assert max(lens) <= MAX_TOK, "512토큰을 넘는 조각이 있습니다 (5-3 기준 2 확인)"
"""),
    CODE("""
# 4. 임베딩 함수: 최대 512토큰 → 모델 → CLS 벡터 → 길이 1로 정규화
@torch.inference_mode()
def encode(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt").to(dev)
        h = model(**enc).last_hidden_state[:, 0]
        out.append(F.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 1024), np.float32)
"""),
    CODE("""
# 5. 정밀(fp32) 기준값: 조각 20개 (7번 셀에서 반정밀 결과와 비교)
random.seed(20261113)
sample_idx = sorted(random.sample(range(len(rows)), min(20, len(rows))))
v32 = encode([rows[i]["embed_text"] for i in sample_idx], bs=8)
print("fp32 기준 벡터", v32.shape)
"""),
    CODE("""
# 6. 전부 임베딩 (GPU면 반정밀 fp16 으로 빠르게). 길이순으로 묶어 패딩을 줄이고, 끝나면 원래 순서로
if dev == "cuda":
    model.half()
order = sorted(range(len(rows)), key=lambda i: lens[i])
t0 = time.time()
vs = encode([rows[i]["embed_text"] for i in order], bs=64 if dev == "cuda" else 8)
vecs = np.zeros_like(vs)
vecs[order] = vs
elapsed = time.time() - t0
print(f"임베딩 {len(rows):,}조각 · {elapsed / 60:.1f}분 · 정밀도 {'fp16' if dev == 'cuda' else 'fp32'}")
"""),
    CODE("""
# 7. 확인 (기준 1·3·4)
norms = np.linalg.norm(vecs, axis=1)
fp_cos = (vecs[sample_idx] * v32).sum(1)
check = {
    "벡터 수": int(vecs.shape[0]), "조각 수": len(rows), "차원": int(vecs.shape[1]),
    "NaN": int(np.isnan(vecs).sum()), "0벡터": int((norms < 1e-6).sum()),
    "길이 최소": round(float(norms.min()), 5), "길이 최대": round(float(norms.max()), 5),
    "fp16 vs fp32 코사인 최소": round(float(fp_cos.min()), 6), "fp16 vs fp32 코사인 평균": round(float(fp_cos.mean()), 6),
    "같은 벡터(중복) 쌍 수": int(len(rows) - len({v.tobytes() for v in vecs.round(5)})),
}
for k, v in check.items():
    print(f"  {k}: {v}")
assert check["벡터 수"] == check["조각 수"] and check["차원"] == 1024
assert check["NaN"] == 0 and check["0벡터"] == 0
assert abs(check["길이 최소"] - 1) <= 1e-3 and abs(check["길이 최대"] - 1) <= 1e-3
assert check["fp16 vs fp32 코사인 최소"] >= 0.999, "반정밀 결과가 정밀과 다릅니다"
print("확인 통과")
# 같은 벡터 쌍은 서로 다른 법령의 같은 문장(예: '감독원장은 이 규정의 시행에 필요한 세부사항을 정할 수 있다.')이면 정상
"""),
    CODE("""
# 8. 맛보기 검색 (기준 6, 눈으로): 예시 질문 → 1~5위 조각 머리말
qv = encode(manifest["questions"], bs=16)
sims = qv @ vecs.T
for q, s in zip(manifest["questions"], sims):
    print("\\nQ.", q)
    for j in np.argsort(-s)[:5]:
        print(f"   {s[j]:.3f}  {rows[j]['embed_text'].splitlines()[0][:110]}")
"""),
    CODE("""
# 9. 저장 + 내려받기: kure_vectors.npz (chunk_id, input_hash, vectors float32, 확인값)
meta = {"model": MODEL, "dtype": "fp16" if dev == "cuda" else "fp32", "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "made": time.strftime("%Y-%m-%d %H:%M"), "file_sha256": manifest["file_sha256"], "chunk_rule": manifest["chunk_rule"],
        "max_tokens": MAX_TOK, "pooling": "cls+normalize", "minutes": round(elapsed / 60, 2), "check": check,
        "torch": torch.__version__, "local_test": LOCAL_TEST}
OUT = os.environ.get("OUT_NPZ", "kure_vectors.npz")
np.savez_compressed(OUT, chunk_id=np.array([r["chunk_id"] for r in rows]), input_hash=np.array([r["input_hash"] for r in rows]),
                    vectors=vecs.astype(np.float32), sample_idx=np.array(sample_idx), sample_fp32=v32,
                    meta=np.array(json.dumps(meta, ensure_ascii=False)))
print(OUT, f"{os.path.getsize(OUT) / 1e6:.0f}MB")
if IN_COLAB:
    files.download(OUT)
"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT, len(cells), "칸")
