"""02 상세: 상세 페이지 HTML을 받아 둔다. 받은 파일은 건너뛰므로 끊겨도 이어서 받는다.
출력: raw/detail/{case_key}.html, raw/detail_failed.json
"""
import json

import common
from common import BASE, RAW, manifest, read_jsonl, request


def url_of(row):
    """포털 원문 링크 (수집과 답변 표시에 함께 쓴다)."""
    if row["gubun"] == "법령해석":
        return f"https://better.fsc.go.kr/fsc_new/replyCase/LawreqDetail.do?stNo=11&muNo=117&muGpNo=75&lawreqIdx={row['idx']}"
    return f"https://better.fsc.go.kr/fsc_new/replyCase/OpinionDetail.do?stNo=11&muNo=117&muGpNo=75&opinionIdx={row['idx']}"


def main():
    out = RAW / "detail"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl("case_list.jsonl")
    new, skip, failed = 0, 0, []
    for r in rows:
        path = out / f"{r['case_key']}.html"
        if path.exists() and path.stat().st_size > 5000:
            skip += 1
            continue
        try:
            resp = request("GET", url_of(r))
            resp.encoding = "utf-8"
            if "질의요지" not in resp.text:
                raise ValueError("상세 본문 표가 없음")
            path.write_text(resp.text, encoding="utf-8")
            new += 1
        except Exception as e:      # 한 건 실패로 전체를 멈추지 않는다
            failed.append({"key": r["case_key"], "error": str(e)[:200]})
    (RAW / "detail_failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest("02_details", target=len(rows), new=new, skipped=skip, failed=len(failed), requests=common.n_requests)


if __name__ == "__main__":
    main()
