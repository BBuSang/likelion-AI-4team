"""무인 파이프라인 탐지 결과(detect.jsonl: {id, spans}) → 1주차 채점기 run_compare.py 예측 형식({id, entities})

사용법:
    python scripts/to_compare_pred.py outputs/holdout2_first/detect.jsonl outputs/holdout2_first/pred_compare.jsonl
    python ../../실전프로젝트1/run_compare.py --gold 정답.jsonl --pred pipeline=outputs/holdout2_first/pred_compare.jsonl --out 결과폴더
"""
import json
import sys


def main(src, dst):
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ents = [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in r.get("spans", [])]
            g.write(json.dumps({"id": r["id"], "entities": ents}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n}건 → {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
