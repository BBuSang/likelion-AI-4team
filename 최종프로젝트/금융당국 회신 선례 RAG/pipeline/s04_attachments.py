"""04 첨부: 본문이 첨부에만 있는 건의 hwp/hwpx를 받아 글을 뽑고 섹션으로 나눈다. 받은 파일은 건너뛴다.
입력: processed/cases_parsed.jsonl
출력: raw/attachments/, raw/attachments.json, processed/cases_full.jsonl, processed/attachment_errors.json
"""
import json
import re
from urllib.parse import quote

import common
from common import BASE, OUT, RAW, SECS, manifest, read_jsonl, request, write_jsonl
from hwp_text import to_text
from text_rules import clean

HEAD = {
    "question": r"질\s*의\s*요\s*지|질\s*의\s*내\s*용|질\s*의\s*사\s*항|요\s*청\s*내\s*용|(?:요청)?대상\s*행위",
    "answer": r"회\s*답|판\s*단",
    "reason": r"이\s*유|판\s*단\s*이\s*유|판\s*단\s*근\s*거|검\s*토\s*이\s*유",
}
HEAD_LINE = re.compile(
    r"^[ \t]*(?:[0-9]+[.)]|[□■◎○●▣◇◆【\[<]|[ⅠⅡⅢ]+\.)?[ \t]*(?:"
    + "|".join(f"(?P<{k}>{v})" for k, v in HEAD.items()) + r")[ \t]*[】\]>]?[ \t]*[:：]?[ \t]*$", re.M)
# 비조치 회신문 머리 "(☑비조치 □조치 □기타)". 체크 기호가 특수 글꼴 문자이거나 사라진 경우가 많아
# "빈 칸 기호가 앞에 없는 선택지"를 체크된 값으로 본다. 정확히 하나일 때만 인정.
CHECKBOX = re.compile(r"\(([^()\n]{0,6}비\s*조\s*치[^()\n]{0,6}조\s*치[^()\n]{0,6}기\s*타)\s*\)")
UNCHECKED = set("□☐\U000f02eb")


def checked_type(head):
    m = CHECKBOX.search(head)
    if not m:
        return None, None
    inner = m.group(1)
    picks = []
    for opt in ("비조치", "조치", "기타"):
        for mm in re.finditer(r"\s*".join(opt), inner):
            if opt == "조치" and inner[max(0, mm.start() - 1):mm.start()] == "비":
                continue  # '비조치' 안의 '조치'
            prev = inner[:mm.start()].rstrip()[-1:] if inner[:mm.start()].strip() else ""
            if prev not in UNCHECKED:
                picks.append(opt)
            break
    return (picks[0] if len(picks) == 1 else None), m.group(0)
TAIL = re.compile(r"\n[ \t]*※[ \t]*(비조치의견서|법령해석)의[ \t]*효력[\s\S]*$")


def split_attachment(text):
    """섹션 제목 줄로 나눈다. 첫 제목 앞(서식 번호·담당자 이름·연락처)은 버린다."""
    # 표 칸 안에서 제목이 "요청대상 / 행위" 또는 "요청 / 대상 / 행위"로 줄이 나뉜 양식이 있다
    text = re.sub(r"^[ \t]*요[ \t]*청[ \t]*\n?[ \t]*대[ \t]*상[ \t]*\n?[ \t]*행[ \t]*위[ \t]*$", "요청대상 행위", text, flags=re.M)
    # "판단 / 이유" 두 줄로 나뉜 '판단이유' 제목
    text = re.sub(r"^[ \t]*판[ \t]*단[ \t]*\n[ \t]*이[ \t]*유[ \t]*$", "판단이유", text, flags=re.M)
    text = TAIL.sub("", text)
    marks = []
    for m in HEAD_LINE.finditer(text):
        kind = next(k for k in HEAD if m.group(k))
        if kind == "answer" and re.fullmatch(r"[^가-힣]*판\s*단\s*이\s*유[^가-힣]*", m.group(0).strip()):
            kind = "reason"   # 정규식 순서상 '판단'이 먼저 잡히는 것을 막음 (안전장치)
        if all(kind != x[0] for x in marks):
            marks.append((kind, m.start(), m.end()))
    if not marks:
        return None
    out = {}
    for i, (kind, _, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        out[kind] = text[end:stop].strip()
    return out


def fetch_files(key, att):
    """첨부 파일을 받아 두고 목록 항목을 돌려준다. 포털 부담을 줄이려고 간격은 2초."""
    out = RAW / "attachments"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for a in att["links"]:
        name = a["orgFileName"]
        safe = re.sub(r'[\\/:*?"<>|]', "_", name)
        local = out / f"{key}__{safe}"
        entry = {"file": local.name, "orgFileName": name, "ext": name.rsplit(".", 1)[-1].lower()}
        if not (local.exists() and local.stat().st_size > 0):
            url = (f"{BASE}/file/displayFile.do?filePath={quote(a['filePath'])}"
                   f"&orgFileName={quote(name)}&sysFileName={quote(a['sysFileName'])}")
            try:
                resp = request("GET", url, delay=2.0)
                if len(resp.content) < 100 or resp.content[:15].lower().startswith(b"<!doctype html"):
                    raise ValueError("파일이 아니라 HTML이 옴")
                local.write_bytes(resp.content)
            except Exception as e:
                entry["error"] = str(e)[:200]
        files.append(entry)
    return files


def main():
    rows = read_jsonl("cases_parsed.jsonl")
    index, att_log = {}, []
    for rec in rows:
        att = rec.pop("_att")
        if not att["need"]:
            continue
        key = rec["case_key"]
        files = fetch_files(key, att)
        index[key] = {"reason": att["reason"], "body_chars": att["body_chars"], "files": files}
        texts = []
        for f in files:
            try:
                texts.append(to_text(RAW / "attachments" / f["file"]))
            except Exception as e:
                att_log.append({"case_key": key, "file": f["file"], "error": str(e)[:150]})
        if not texts:
            rec["notes"].append("첨부 글 추출 불가 (" + ",".join(f["ext"] for f in files) + ")")
            continue
        joined = "\n".join(texts)
        if not rec["answer_type"]:
            typ, raw_box = checked_type(joined[:600])
            if raw_box:
                rec["answer_type"], rec["answer_type_raw"] = typ, raw_box
                if typ is None:
                    rec["notes"].append("답변유형 체크칸 판독 불가")
        secs = split_attachment(joined)
        if secs:
            for s in SECS:
                if secs.get(s, "").strip():
                    rec[s] = secs[s]
            rec["body_source"] = "attachment"
            missing = [s for s in SECS if not secs.get(s, "").strip()]
            if missing:
                rec["notes"].append(f"첨부에 섹션 없음: {','.join(missing)}")
        else:
            rec["body_source"] = "attachment_unsplit"
            if sum(len(clean(rec[s])) for s in SECS) < 200:
                rec["reason"] = joined
            rec["notes"].append("첨부 섹션 분리 실패")
    (RAW / "attachments.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "attachment_errors.json").write_text(json.dumps(att_log, ensure_ascii=False, indent=1), encoding="utf-8")
    write_jsonl("cases_full.jsonl", rows)
    manifest("04_attachments", need=len(index), text_errors=len(att_log), requests=common.n_requests,
             unsplit=sum(r["body_source"] == "attachment_unsplit" for r in rows))


if __name__ == "__main__":
    main()
