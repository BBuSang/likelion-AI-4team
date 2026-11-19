"""기획안 5-4 코랩 입력 묶음: data/work/05_embed/kure_input.zip

    kure_input.zip
      chunks_input.jsonl   (chunk_id, embed_text, input_hash)  ← 5-3 결과 그대로
      manifest.json        (조각 수, 파일 지문, 규칙 버전, 예시 질문)

    python pipeline/colab_bundle_04.py
"""
import hashlib
import json
import zipfile

from common import TODAY, WORK

SRC = WORK / "04_chunks" / "chunks_input.jsonl"
OUT = WORK / "05_embed"

# 기획안 2-1 예시 질문 (코랩에서 검색 맛보기용)
QUESTIONS = [
    "은행이 고객에게 금융상품을 판매할 때 설명해야 하는 의무는 어떤 법에 있어?",
    "금융회사가 고객 신용정보를 제3자에게 제공하려면 어떤 조건이 필요해?",
    "전자금융사고가 나면 손해배상 책임은 누가 져?",
    "예금자보호 한도는 1인당 얼마야?",
    "대부업자가 받을 수 있는 최고 이자율은?",
    "고객 금융거래 정보를 수사기관에 줄 수 있는 경우는?",
    "자금세탁이 의심되는 거래는 어디에 언제까지 보고해야 해?",
    "보험 청약을 철회할 수 있는 기간은?",
    "개인정보가 유출되면 정보주체에게 언제 알려야 해?",
    "가상자산사업자는 이용자 예치금을 어떻게 보관해야 해?",
    "은행이 대주주에게 해 줄 수 있는 신용공여 한도는?",
    "신용카드 회원을 모집할 때 금지되는 행위는?",
]


def main():
    raw = SRC.read_bytes()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    stats = json.loads((WORK / "04_chunks" / "chunk_stats.json").read_text(encoding="utf-8"))
    manifest = {
        "made_on": TODAY, "n_chunks": len(rows), "file_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_rule": stats["rule"], "model": "nlpai-lab/KURE-v1", "max_tokens": 512, "questions": QUESTIONS,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT / "kure_input.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("chunks_input.jsonl", raw)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("kure_input.zip", round((OUT / "kure_input.zip").stat().st_size / 1e6, 1), "MB · 조각", len(rows),
          "· 지문", manifest["file_sha256"][:12])


if __name__ == "__main__":
    main()
