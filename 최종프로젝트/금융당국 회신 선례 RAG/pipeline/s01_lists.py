"""01 목록: 최근회신사례 + 통합조회. 이미 받은 목록은 다시 요청하지 않는다 (--refresh면 새로 받음).
출력: raw/list_recent.json, raw/list_total.json, processed/case_list.jsonl
"""
import json
import sys

from common import BASE, RAW, case_key, manifest, request, write_jsonl

KEEP = ("법령해석", "비조치의견서")


def get_list(name, path):
    p = RAW / f"{name}.json"
    if p.exists() and "--refresh" not in sys.argv:
        return json.loads(p.read_text(encoding="utf-8"))["data"]
    data = request("POST", f"{BASE}/replyCase/{path}", data={"draw": 1, "start": 0, "length": 20000}).json()
    assert len(data["data"]) == data["recordsTotal"], f"{name}: 일부만 받음"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["data"]


def main():
    recent = get_list("list_recent", "selectReplyCasePastReplyList.do")
    total = {(t["pastreqType"], t["dataIdx"]): t.get("replyRegDate")
             for t in get_list("list_total", "selectReplyCaseTotalReplyList.do")}
    rows = [{
        "case_key": case_key(r["gubun"], r["idx"]), "gubun": r["gubun"], "idx": r["idx"], "serial_no": r["number"],
        "title": r["title"].strip(), "category": r["category"] or None,
        "reg_date": r["regDate"].replace(".", "-") if r.get("regDate") else None,
        "reply_date_list": total.get((r["gubun"], r["idx"])),   # 통합조회 회신일 (대조·보정용)
    } for r in recent if r["gubun"] in KEEP]
    write_jsonl("case_list.jsonl", rows)
    manifest("01_lists", cases=len(rows), no_total_date=sum(r["reply_date_list"] is None for r in rows))


if __name__ == "__main__":
    main()
