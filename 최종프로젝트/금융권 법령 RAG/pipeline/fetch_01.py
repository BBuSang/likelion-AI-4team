"""기획안 5-1 데이터 수집

원본(data/source/api)을 캐시로 쓰고, 기준일 현행판이 아니거나 빠진 것만 API로 새로 받는다.
새로 받은 것은 data/work/01_fetch/api/ 에 둔다 (원본은 덮어쓰지 않음).

    python pipeline/fetch_01.py inventory   # 원본 파일 목록·범위 분류
    python pipeline/fetch_01.py check       # API로 현행판·하위 법령 확인 (1초 간격, 캐시)
    python pipeline/fetch_01.py fetch       # 빠진 것·바뀐 판 받기
    python pipeline/fetch_01.py export      # 최종 목록 + CSV (laws.csv, articles_raw.csv)
    python pipeline/fetch_01.py all
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import SOURCE, TODAY, WORK, api_get, as_list, content

OUT = WORK / "01_fetch"
CACHE = OUT / "cache"
LOG = OUT / "fetch_log.jsonl"
NEW_API = OUT / "api"
RULE_VERSION = "5-1 v2 (기획안 v0.5)"

ADMRUL_IN = {"고시", "세칙"}  # 기획안 3-2: 행정규칙은 고시·세칙만
SUBS = (" 시행령", " 시행규칙")


# ---------- 파일 읽기 ----------
def law_level(kind):
    if kind == "법률":
        return "법률"
    if kind == "대통령령":
        return "시행령"
    if kind and kind.endswith("령"):  # 총리령, ○○부령
        return "시행규칙"
    return "기타"  # 대법원규칙 등


def read_file(target, path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if target == "law":
        L = d["법령"]
        b = L["기본정보"]
        kind = content(b.get("법종구분"))
        units = as_list((L.get("조문") or {}).get("조문단위"))
        return {
            "target": "law", "uid": f"law:{Path(path).stem}", "mst": Path(path).stem, "law_id": b.get("법령ID"),
            "name": b.get("법령명_한글"), "abbrev": b.get("법령명약칭") or "", "kind_raw": kind,
            "level": law_level(kind), "org": content(b.get("소관부처")),
            "effective": b.get("시행일자"), "promulgation": b.get("공포일자"),
            "n_units": len(units), "n_articles": sum(u.get("조문여부") == "조문" for u in units),
            "n_chapters": sum(u.get("조문여부") == "전문" for u in units),
            "has_annex": bool(L.get("별표")), "path": str(path),
        }
    A = d["AdmRulService"]
    b = A["행정규칙기본정보"]
    body = A.get("조문내용")
    shape = "list" if isinstance(body, list) else ("str" if isinstance(body, str) else "none")
    items = as_list(body) if shape == "list" else []
    return {
        "target": "admrul", "uid": f"admrul:{b.get('행정규칙일련번호')}", "mst": b.get("행정규칙일련번호"),
        "law_id": b.get("행정규칙ID"), "name": b.get("행정규칙명"), "abbrev": "", "kind_raw": b.get("행정규칙종류"),
        "level": "행정규칙", "org": b.get("소관부처명"), "effective": b.get("시행일자"),
        "promulgation": b.get("발령일자"), "body_shape": shape,
        "n_units": len(items) if shape == "list" else (1 if shape == "str" else 0),
        "n_articles": sum(bool(ADM_ART.match(str(x).strip())) for x in items),
        "has_annex": bool(A.get("별표")), "current_flag": b.get("현행여부"), "path": str(path),
    }


ADM_ART = re.compile(r"^제\s*(\d+(?:-\d+)?)\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^)]*)\))?")


def scan(api_dir):
    out = []
    for target in ("law", "admrul"):
        for p in sorted((Path(api_dir) / target).glob("*.json")):
            out.append(read_file(target, p))
    return out


# ---------- 범위 ----------
def target_laws():
    t = json.loads((SOURCE / "target_laws.json").read_text(encoding="utf-8"))
    rows = []
    for g, d in t.items():
        if g.startswith("_"):
            continue
        for short, (full, n) in d.items():
            rows.append({"group": g, "short": short, "name": full, "n_cases": n})
    return rows


def scope_of(r, a_names):
    if r["target"] == "admrul":
        return "C" if r["kind_raw"] in ADMRUL_IN else "제외"
    if r["level"] == "기타":  # 대법원규칙 등 (기획안 3-2: 법원인사사무규칙 제외)
        return "제외"
    if r["name"] in a_names:
        return "A"
    if any(r["name"] == a + s for a in a_names for s in SUBS):
        return "B"
    return "D"


def inventory():
    src = scan(SOURCE / "api")
    for r in src:
        r["origin"] = "source"
    fut = [dict(read_file("law", p), origin="source_future") for p in sorted((SOURCE / "api" / "law_future").glob("*.json"))]
    a_names = {t["name"] for t in target_laws()}
    for r in src + fut:
        r["scope"] = scope_of(r, a_names)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(src + fut, ensure_ascii=False, indent=1), encoding="utf-8")
    c = Counter((r["origin"], r["target"], r["scope"]) for r in src + fut)
    print("원본 파일:", len(src), "+ 예정판", len(fut))
    for k, v in sorted(c.items()):
        print("  ", k, v)
    return src + fut


# ---------- API 확인 ----------
def search_law(name):
    res = api_get("lawSearch.do", CACHE / "search", LOG, target="law", query=name, nw="3", display=100)
    return as_list(res.get("LawSearch", {}).get("law"))


def check():
    inv = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    targets = target_laws()
    a_names = [t["name"] for t in targets]
    # 확인할 법령 이름: 범위 안(A·B·D) 법령 + A 법률의 시행령·시행규칙 이름
    names = {r["name"] for r in inv if r["target"] == "law" and r["scope"] in ("A", "B", "D")}
    names |= {a + s for a in a_names for s in [""] + list(SUBS)}
    current = {}
    for i, nm in enumerate(sorted(names), 1):
        hits = [x for x in search_law(nm) if x.get("법령명한글") == nm]
        current[nm] = [{"mst": x.get("법령일련번호"), "law_id": x.get("법령ID"), "effective": x.get("시행일자"),
                        "promulgation": x.get("공포일자"), "kind": x.get("법령구분명"),
                        "status": x.get("현행연혁코드")} for x in hits]
        if i % 50 == 0:
            print(f"  법령 {i}/{len(names)}")
    # 행정규칙: 행정규칙ID(LID)로 현행판 본문을 바로 받는다 (이름 검색보다 확실)
    adm = {}
    lids = sorted({r["law_id"] for r in inv if r["target"] == "admrul" and r["scope"] == "C"})
    for i, lid in enumerate(lids, 1):
        body = api_get("lawService.do", NEW_API / "_admrul_lid", LOG, target="admrul", LID=lid)
        b = (body.get("AdmRulService") or {}).get("행정규칙기본정보") or {}
        adm[lid] = {"serial": b.get("행정규칙일련번호"), "name": b.get("행정규칙명"), "effective": b.get("시행일자"),
                    "kind": b.get("행정규칙종류"), "current_flag": b.get("현행여부"), "ok": bool(b)}
        if i % 50 == 0:
            print(f"  행정규칙 {i}/{len(lids)}")
    (OUT / "api_current.json").write_text(json.dumps({"checked_on": TODAY, "law": current, "admrul": adm},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print("법령 이름", len(names), "· 현행 못 찾음", sum(not v for v in current.values()),
          "| 행정규칙", len(lids), "· 못 받음", sum(not v["ok"] for v in adm.values()))


# ---------- 받기 ----------
def fetch():
    inv = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    cur = json.loads((OUT / "api_current.json").read_text(encoding="utf-8"))
    have = {r["mst"]: r["effective"] for r in inv if r["target"] == "law" and r["origin"] == "source"}
    want = []
    for nm, hits in cur["law"].items():
        live = [h for h in hits if (h["effective"] or "0") <= TODAY]
        if len(live) != 1:
            continue  # 0 = 현행 없음(폐지·이름 다름), 2+ = export 에서 사유 표시
        h = live[0]
        # 판(MST)이 없거나, 같은 판이라도 시행일자가 다르면 (단계적 시행: 늦게 시행된 조문이 빠져 있을 수 있음) 다시 받음
        if h["mst"] not in have or have[h["mst"]] != h["effective"]:
            want.append((nm, h))
    print("새로 받을 법령:", len(want))
    (NEW_API / "law").mkdir(parents=True, exist_ok=True)
    for nm, h in want:
        p = NEW_API / "law" / f"{h['mst']}.json"
        if p.exists():
            continue
        body = api_get("lawService.do", CACHE / "body", LOG, target="eflaw", MST=h["mst"], efYd=h["effective"])
        if "법령" not in body:
            print("  ⚠ 못 받음:", nm, str(body)[:80])
            continue
        p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        print("  받음:", nm, h["effective"])
    # 행정규칙: LID 응답이 원본과 다른 판이면 그 응답을 새 파일로
    src_serials = {r["mst"]: r["effective"] for r in inv if r["target"] == "admrul"}
    (NEW_API / "admrul").mkdir(parents=True, exist_ok=True)
    n = 0
    for lid, a in cur["admrul"].items():
        if a["ok"] and (a["serial"] not in src_serials or src_serials[a["serial"]] != a["effective"]):
            body = api_get("lawService.do", NEW_API / "_admrul_lid", LOG, target="admrul", LID=lid)
            (NEW_API / "admrul" / f"{a['serial']}.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            n += 1
            print("  행정규칙 새 판:", a["name"], a["effective"])
    print("새로 받은 행정규칙:", n)


# ---------- 최종 목록 + CSV ----------
def article_label(no, sub):
    return f"제{no}조" + (f"의{sub}" if sub and str(sub) not in ("0", "") else "")


def _texts(x):
    return [str(t) for t in as_list(x) if str(t).strip()] if not isinstance(x, list) else \
        [t for y in x for t in _texts(y)]


def law_units(path):
    L = json.loads(Path(path).read_text(encoding="utf-8"))["법령"]
    for u in as_list((L.get("조문") or {}).get("조문단위")):
        lines = _texts(u.get("조문내용"))
        for h in as_list(u.get("항")):
            lines += _texts(h.get("항내용"))
            for ho in as_list(h.get("호")):
                lines += _texts(ho.get("호내용"))
                for m in as_list(ho.get("목")):
                    lines += _texts(m.get("목내용"))
        yield {
            "unit_key": u.get("조문키"), "unit_type": u.get("조문여부"),
            "article": article_label(u.get("조문번호"), u.get("조문가지번호")) if u.get("조문여부") == "조문" else "",
            "title": u.get("조문제목") or "", "unit_effective": u.get("조문시행일자") or "",
            "raw_text": "\n".join(s.strip() for s in lines),
        }


def admrul_units(path):
    A = json.loads(Path(path).read_text(encoding="utf-8"))["AdmRulService"]
    body = A.get("조문내용")
    if isinstance(body, str):
        yield {"unit_key": "1", "unit_type": "본문(분리 전)", "article": "", "title": "", "unit_effective": "",
               "raw_text": body.strip()}
        return
    for i, x in enumerate(as_list(body), 1):
        s = str(x).strip()
        m = ADM_ART.match(s)
        yield {"unit_key": str(i), "unit_type": "조문" if m else "기타(장·절 등)",
               "article": article_label(m.group(1), m.group(2)) if m else "", "title": (m.group(3) or "") if m else "",
               "unit_effective": "", "raw_text": s}


def export():
    inv = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    cur = json.loads((OUT / "api_current.json").read_text(encoding="utf-8"))
    new = scan(NEW_API)
    for r in new:
        r["origin"] = "api_1113"
    a_names = {t["name"] for t in target_laws()}
    for r in new:
        r["scope"] = scope_of(r, a_names)
    allf = inv + new
    by_mst = {r["mst"]: r for r in allf if r["origin"] != "source_future"}

    chosen, dropped = [], []
    # 법령: 이름마다 API 현행판 하나
    law_names = {r["name"] for r in allf if r["target"] == "law" and r["scope"] in ("A", "B", "D")} | set(cur["law"])
    for nm in sorted(law_names):
        hits = [h for h in cur["law"].get(nm, []) if (h["effective"] or "0") <= TODAY]
        files = [r for r in allf if r["target"] == "law" and r["name"] == nm and r["origin"] != "source_future"]
        if files and all(f["scope"] == "제외" for f in files):
            dropped += [{**f, "why": "범위 밖: " + str(f["kind_raw"])} for f in files]
            continue
        if len(hits) == 1 and hits[0]["mst"] in by_mst:
            c = dict(by_mst[hits[0]["mst"]])
            c["api_effective"] = hits[0]["effective"]
            chosen.append(c)
            for f in files:
                if f["path"] != c["path"]:
                    dropped.append({**f, "why": "현행판 아님 (같은 법령의 다른 판)" if f["mst"] != c["mst"]
                                    else "같은 판, 기준일 시행일자로 다시 받음"})
        elif not hits and files:
            for f in files:
                dropped.append({**f, "why": "API 현행 목록에 없음 (폐지·이름 변경 확인 필요)"})
        elif len(hits) > 1:
            for f in files:
                dropped.append({**f, "why": f"현행판이 {len(hits)}개 (확인 필요)"})
        elif hits:
            dropped.append({"name": nm, "target": "law", "why": "현행판을 받지 못함", "scope": "?"})
        # hits 없음 + 파일 없음 = A 의 시행령·시행규칙 이름 후보가 실제로 없는 경우 → 정상
    # 행정규칙: 고시·세칙만, LID 현행판
    for lid, a in cur["admrul"].items():
        files = [r for r in allf if r["target"] == "admrul" and r["law_id"] == lid]
        if a["ok"] and a["serial"] in by_mst and (a["effective"] or "0") <= TODAY:
            c = dict(by_mst[a["serial"]])
            c["api_effective"] = a["effective"]
            chosen.append(c)
            dropped += [{**f, "why": "현행판 아님 (같은 규정의 다른 판)" if f["mst"] != a["serial"]
                         else "같은 판, 기준일 시행일자로 다시 받음"} for f in files if f["path"] != c["path"]]
        else:
            dropped += [{**f, "why": f"현행판 확인 실패 ({a})"} for f in files]
    dropped += [{**r, "why": "범위 밖: " + str(r["kind_raw"])} for r in allf if r["target"] == "admrul" and r["scope"] == "제외"]
    dropped += [{**r, "why": "시행 예정판 (보관만)"} for r in allf if r["origin"] == "source_future"]

    # 상위 법률 연결 (시행령·시행규칙 → 이름에서 뒤 꼬리 뗀 법률)
    names = {c["name"]: c["uid"] for c in chosen}
    for c in chosen:
        base = next((c["name"][: -len(s)] for s in SUBS if c["name"].endswith(s)), None)
        c["parent_uid"] = names.get(base, "") if base else ""

    (OUT / "selected.json").write_text(json.dumps({"checked_on": TODAY, "rule": RULE_VERSION, "chosen": chosen,
                                                   "dropped": dropped}, ensure_ascii=False, indent=1), encoding="utf-8")
    nc = {t["name"]: (t["group"], t["n_cases"]) for t in target_laws()}
    cols = ["uid", "scope", "law_name", "level", "kind_raw", "law_id", "mst", "abbrev", "org", "effective",
            "promulgation", "parent_uid", "sector", "n_cases", "n_units", "n_articles", "origin", "file"]
    with open(OUT / "laws.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in sorted(chosen, key=lambda c: (c["scope"], c["name"])):
            g, n = nc.get(c["name"], ("", ""))
            w.writerow([c["uid"], c["scope"], c["name"], c["level"], c["kind_raw"], c["law_id"], c["mst"], c["abbrev"],
                        c["org"], c["effective"], c["promulgation"], c["parent_uid"], g, n, c["n_units"],
                        c["n_articles"], c["origin"], Path(c["path"]).relative_to(SOURCE.parents[1]).as_posix()])
    cols = ["uid", "law_name", "level", "scope", "unit_key", "unit_type", "article", "title", "raw_text",
            "unit_effective", "law_effective"]
    n_rows = 0
    with open(OUT / "articles_raw.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in sorted(chosen, key=lambda c: (c["scope"], c["name"])):
            units = law_units(c["path"]) if c["target"] == "law" else admrul_units(c["path"])
            for u in units:
                w.writerow([c["uid"], c["name"], c["level"], c["scope"], u["unit_key"], u["unit_type"], u["article"],
                            u["title"], u["raw_text"], u["unit_effective"], c["effective"]])
                n_rows += 1
    print("최종 법령", len(chosen), Counter((c["scope"], c["level"]) for c in chosen))
    print("뺀 파일", len(dropped), Counter(d["why"].split(" (")[0].split(":")[0] for d in dropped))
    print("articles_raw.csv 행", n_rows)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for s in (["inventory", "check", "fetch", "export"] if step == "all" else [step]):
        print(f"== {s}")
        globals()[s]()
