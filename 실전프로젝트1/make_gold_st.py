"""성+직함 포함 정답 만들기 (make_gold_st.py).

팀 기준을 바꿔 '김 대리'·'정 부장' 같은 성+직함도 person으로 볼 때 쓰는 정답 파일을 만듭니다.
eda.py의 '성+호칭' 후보(dev 17건, test 40건, 수동 확인)를 정답에 더합니다.
탐지 규칙(detectors/base.py surname_title)과 별개로 eda.py 정규식을 써서 정답을 만듭니다.

사용법
    python make_gold_st.py
출력
    data/pii_synth_finance_dev_st.jsonl, data/pii_synth_finance_test_st.jsonl
"""
import json

from eda import DATA, unlabeled_candidates


def main():
    for split, src in DATA.items():
        out, added = src.replace(".jsonl", "_st.jsonl"), 0
        with open(src, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
            for line in f:
                d = json.loads(line)
                for kind, text, s, e in unlabeled_candidates(d):
                    if kind == "성+호칭":
                        d["entities"].append({"label": "person", "start": s, "end": e, "text": text,
                                              "form": "surname_title", "person_id": None})
                        added += 1
                d["entities"].sort(key=lambda x: x["start"])
                g.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{out}: 성+직함 {added}건 추가")


if __name__ == "__main__":
    main()
