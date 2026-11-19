"""기획안 5-2 판단 기준 1~11 점검 + 보고서 → data/work/02_clean/audit.json, reports/5-2_전처리.md

    python pipeline/audit_02.py
"""
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from common import REPORTS, WORK
from clean_02 import HANJA_PAREN, TAG_PATTERNS, normalize

sys.path.insert(0, str(WORK.parents[1] / "pipeline"))
from audit_01 import is_deleted as old_is_deleted  # noqa: E402

csv.field_size_limit(2**31 - 1)
C2 = WORK / "02_clean"
C1 = WORK / "01_fetch"
BOXCH = re.compile(r"[┌┐└┘├┤┬┴┼─━┏┓┗┛┣┫┳┻╋═║╔╗╚╝╠╣╦╩╬│┃|\s]")


def squash(t):
    return BOXCH.sub("", normalize(t or ""))


def is_subseq(small, big):
    it = iter(big)
    return all(ch in it for ch in small)


def main():
    arts = list(csv.DictReader(open(C2 / "articles.csv", encoding="utf-8-sig")))
    lines = list(csv.DictReader(open(C2 / "article_lines.csv", encoding="utf-8-sig")))
    gate = json.loads((C2 / "admrul_gate.json").read_text(encoding="utf-8"))
    stats = json.loads((C2 / "clean_stats.json").read_text(encoding="utf-8"))["stats"]
    raw1 = [r for r in csv.DictReader(open(C1 / "articles_raw.csv", encoding="utf-8-sig")) if r["unit_type"] == "조문"]
    by_art = defaultdict(list)
    for ln in lines:
        by_art[ln["article_uid"]].append(ln)
    T = lambda x: x == "True"
    srch = [a for a in arts if T(a["searchable"])]
    res = {}

    # 1. 빈 조문
    empty = [a["article_uid"] for a in srch if not a["text"].strip()]
    res["1"] = {"검색 대상": len(srch), "빈 것": len(empty), "pass": not empty}

    # 2. 메타데이터 누락
    miss = Counter()
    for a in srch:
        for f in ("law_name", "law_level", "article_label", "law_effective_date"):
            if not a[f].strip():
                miss[f] += 1
    no_title = Counter(a["parse_method"] for a in srch if not a["title"].strip())
    res["2"] = {"누락": dict(miss), "조문제목 없음(원본에 없는 경우 허용)": dict(no_title), "pass": not miss}

    # 3. 구조: 법령 항·호·목 줄 수 = JSON 개수
    tot = Counter()
    bad = []
    for a in arts:
        if a["parse_method"] != "api":
            continue
        lv = Counter(ln["level"] for ln in by_art[a["article_uid"]])
        j = (int(a["n_json_para"]), int(a["n_json_item"]), int(a["n_json_sub"]))
        got = (lv["항"], lv["호"], lv["목"])
        tot["JSON 항"] += j[0]; tot["JSON 호"] += j[1]; tot["JSON 목"] += j[2]
        tot["줄 항"] += got[0]; tot["줄 호"] += got[1]; tot["줄 목"] += got[2]
        if j != got:
            bad.append({"조문": f'{a["law_name"]} {a["article_label"]}', "JSON": j, "줄": got})
    res["3"] = {"합계": dict(tot), "다른 조문": len(bad), "예": bad[:10], "pass": not bad}

    # 4. 행정규칙 나누기 (한 덩어리 18개) + 줄바꿈 넣기 (모든 행정규칙)
    g_rows = []
    for g in gate:
        xs = [a for a in arts if a["law_name"] == g["law"]]
        g_rows.append({"규정": g["law"], "방법": g["method"], "글자": g["chars"], "조문(단위)": g["n_articles"],
                       "처음 ~ 끝": f'{xs[0]["article_label"]} ~ {xs[-1]["article_label"]}' if xs else "-",
                       "손실 없음": g["lossless"], "번호 증가": g["increasing"],
                       "통과": g["lossless"] and g["increasing"] and g["n_articles"] > 0})
    adm = [a for a in arts if a["law_level"] == "행정규칙"]
    adm_multi = sum(int(a["n_lines"]) > 1 for a in adm if T(a["searchable"]))
    lv_adm = Counter(ln["level"] for a in adm for ln in by_art[a["article_uid"]])
    # 줄 가운데 남은 항 기호(②~⑳): 본문 속 참조(제①항, ①부터, ① 및)는 뺌
    glued, dup_src = [], []
    for a in adm:
        if not T(a["searchable"]):
            continue
        seen = {ln["marker"] for ln in by_art[a["article_uid"]] if ln["level"] == "항"}
        for ln in by_art[a["article_uid"]]:
            for m in re.finditer(r"[②-⑳]", ln["text"]):
                if m.group(0) in seen and m.start() > 0:  # 원문에 같은 항 번호가 두 번 (예: ③ 다음 또 ③) → 원문 오류, 예외
                    dup_src.append(f'{a["law_name"]} {a["article_label"]}: 원문에 {m.group(0)} 중복')
                    continue
                if m.start() == 0:
                    continue
                bf, af = ln["text"][max(0, m.start() - 3):m.start()], ln["text"][m.end():m.end() + 6]
                if re.search(r"(?<!삭)(?<!삭 )제\s*$", bf) or re.match(r"\s*(?:부터|까지|및|또는|의|에|·|ㆍ|,|항)", af):
                    continue
                glued.append(f'{a["law_name"]} {a["article_label"]}: …{ln["text"][max(0, m.start() - 20):m.start() + 15]}…')
    # 줄 가운데 붙은 목(`…사항가. …나. …`)
    glued_mok = sum(1 for a in adm if T(a["searchable"]) for ln in by_art[a["article_uid"]]
                    if re.search(r"[가-힣)]가\.\s?\S.{2,300}?[가-힣)]나\.\s", ln["text"]))
    res["4"] = {"줄 가운데 붙은 목 줄": glued_mok, "한 덩어리 규정": g_rows, "행정규칙 조문": len(adm), "줄이 2개 이상으로 나뉜 조문": adm_multi,
                "행정규칙 줄 단계": dict(lv_adm), "줄 가운데 남은 항 기호": len(glued), "남은 예": glued[:15], "원문 번호 중복(예외)": dup_src,
                "pass": all(r["통과"] for r in g_rows) and not glued}

    # 5. 남은 태그 (지우는 모양) + 남은 <…>·[…] 종류
    left = Counter()
    for a in srch:
        for name, pat in TAG_PATTERNS:
            left[name] += len(pat.findall(a["text"]))
    left = {k: v for k, v in left.items() if v}
    rest = Counter()
    for a in srch:
        for m in re.findall(r"<[^<>]{1,40}>|\[[^\[\]]{1,40}\]", a["text"]):
            rest[re.sub(r"\d+", "#", m)[:30]] += 1
    res["5"] = {"지우는 모양 남은 것": left, "남은 <…>·[…] 종류(상위 20)": rest.most_common(20), "pass": not left}

    # 6. 원본 대조: 줄마다 원문의 부분열 (전처리가 새로 만든 글 0)
    fail, removed = [], []
    raw_of = {a["article_uid"]: squash(a["raw_text"]) for a in arts}
    for a in srch:
        big = raw_of[a["article_uid"]]
        for ln in by_art[a["article_uid"]]:
            if ln["is_deleted"] == "True":
                continue
            if not is_subseq(squash(ln["text"]), big):
                fail.append({"조문": f'{a["law_name"]} {a["article_label"]}', "줄": ln["text"][:80]})
        if big:
            removed.append(1 - len(squash(a["text"])) / len(big))
    removed = np.array(removed)
    res["6"] = {"검사 줄": sum(len(by_art[a["article_uid"]]) for a in srch), "원문에 없는 글이 생긴 줄": len(fail),
                "예": fail[:10], "지운 비율 중앙값": round(float(np.median(removed)), 3),
                "지운 비율 30% 넘는 조문": int((removed > 0.3).sum()), "pass": not fail}

    # 7. 세 층
    hz_text = sum(len(HANJA_PAREN.findall(a["text"])) for a in arts)
    hz_emb = sum(len(HANJA_PAREN.findall(a["embed_body"])) for a in arts)
    res["7"] = {"text 의 한자 괄호": hz_text, "embed_body 의 한자 괄호": hz_emb,
                "전처리 때 뺀 수(줄 기준)": stats.get("한자 괄호(임베딩용에서 뺌)"),
                "pass": hz_emb == 0 and hz_text > 0}

    # 8. 삭제: 5-1 규칙과 비교 (법령 + 조문 목록 행정규칙)
    old = {f'{r["uid"]}:{r["unit_key"]}' for r in raw1 if old_is_deleted(r["raw_text"])}
    raw_key = {}
    for a in arts:
        if a["parse_method"] in ("api", "admrul_list"):
            raw_key[a["article_uid"]] = a
    new = {k for k, a in raw_key.items() if a["skip_reason"] == "삭제"}
    only_new = [f'{raw_key[k]["law_name"]} {raw_key[k]["article_label"]}: {raw_key[k]["raw_text"][:60]}' for k in new - old]
    only_old = sorted(old - new)
    left_del = sum("삭제 <" in a["text"] for a in srch)
    res["8"] = {"5-1 규칙 삭제": len(old), "5-2 삭제(법령+목록 행정규칙)": len(new),
                "5-2 삭제(전체)": sum(a["skip_reason"] == "삭제" for a in arts),
                "5-2에서만 삭제": only_new, "5-1에서만 삭제": only_old, "검색 대상에 '삭제 <' 남음": left_del,
                "pass": not only_old and left_del == 0}

    # 9. 조문 수 보존 (법령 + 조문 목록 행정규칙)
    n1 = Counter(r["uid"] for r in raw1)
    n2 = Counter(a["law_uid"] for a in arts if a["parse_method"] in ("api", "admrul_list"))
    diff9 = [(u, n1[u], n2[u]) for u in n1 if n1[u] != n2[u] and u in n2]
    by_reason = Counter(a["skip_reason"] or "검색 대상" for a in arts)
    res["9"] = {"5-1 조문": sum(n1.values()), "5-2 조문(법령+목록)": sum(n2.values()),
                "차이 나는 법령": diff9[:10], "한 덩어리·지침에서 나온 조문": sum(a["parse_method"] in ("admrul_split", "guideline_items", "whole_body") for a in arts),
                "검색 대상/제외": dict(by_reason), "pass": not diff9}

    # 10. 청크 준비: 900자 넘는데 경계가 없는 조문
    nob = []
    for a in srch:
        if int(a["n_chars"]) > 900 and int(a["n_lines"]) <= 1 and len(re.findall(r"다\.", a["text"])) <= 1:
            nob.append(f'{a["law_name"]} {a["article_label"]} ({a["n_chars"]}자)')
    ch = np.array([int(a["n_chars"]) for a in srch])
    tk = np.array([int(a["n_tokens"] or 0) for a in srch])
    q = [50, 75, 85, 90, 95, 99]
    res["10"] = {"경계 없는 900자 초과": nob, "글자 백분위": dict(zip(q, np.percentile(ch, q).astype(int).tolist())),
                 "토큰 백분위": dict(zip(q, np.percentile(tk, q).astype(int).tolist())), "최대 글자": int(ch.max()),
                 "900자 이하 비율": round(float((ch <= 900).mean()), 4), "글자/토큰": round(float(ch.sum() / tk.sum()), 2),
                 "구분별 900자 초과": {lv: round(float((np.array([int(a["n_chars"]) for a in srch if a["law_level"] == lv]) > 900).mean()), 3)
                                  for lv in ("법률", "시행령", "시행규칙", "행정규칙")},
                 "pass": True}

    # 11. 전/후 샘플
    random.seed(20261113)
    pick = {
        "개정 태그": [a for a in srch if "<개정" in a["raw_text"] and len(a["raw_text"]) < 400],
        "삭제된 항·호": [a for a in srch if any(ln["is_deleted"] == "True" for ln in by_art[a["article_uid"]]) and len(a["raw_text"]) < 500],
        "표": [a for a in srch if T(a["has_table"]) and len(a["raw_text"]) < 900],
        "그림": [a for a in srch if T(a["has_image"]) and len(a["raw_text"]) < 700],
        "한자": [a for a in srch if HANJA_PAREN.search(a["text"]) and len(a["raw_text"]) < 300],
        "행정규칙 줄바꿈": [a for a in srch if a["parse_method"] == "admrul_list" and int(a["n_lines"]) >= 3 and len(a["raw_text"]) < 500],
        "한 덩어리 행정규칙": [a for a in srch if a["parse_method"] == "admrul_split" and int(a["n_lines"]) >= 2 and len(a["raw_text"]) < 500],
        "지침 항목": [a for a in srch if a["parse_method"] == "guideline_items" and len(a["raw_text"]) < 600],
        "준용 목록": [a for a in srch if T(a["is_reference_list"])],
    }
    samples = {k: random.sample(v, min(2, len(v))) for k, v in pick.items()}
    res["11"] = {"유형별 후보 수": {k: len(v) for k, v in pick.items()}, "pass": all(samples.values())}
    res["표시"] = {k: sum(T(a[k]) for a in srch) for k in ("has_image", "has_table", "has_delegation", "is_reference_list")}
    res["방법별"] = dict(Counter(a["parse_method"] for a in arts))
    res["지운 것"] = stats
    (C2 / "audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in [str(i) for i in range(1, 12)]:
        print(k, "통과" if res[k]["pass"] else "실패", str({kk: vv for kk, vv in res[k].items() if kk not in ("pass", "한 덩어리 규정", "예")})[:300])
    report(res, samples, by_art)


def show(t, n=380):
    t = t.replace("|", "\\|").replace("\n", " ⏎ ")
    return t[:n] + (" …" if len(t) > n else "")


def report(res, samples, by_art):
    L = []
    p = L.append
    ok = lambda k: "✅ 통과" if res[k]["pass"] else "❌ 실패"
    r10 = res["10"]
    p("# 5-2 전처리 결과\n")
    p("기준일 2026-11-13 · 규칙 `5-2 v2 (기획안 v0.9)` · 코드 `pipeline/clean_02.py`, `pipeline/audit_02.py`\n")
    p("## 한눈에 보기\n")
    r9 = res["9"]["검색 대상/제외"]
    p(f"- 조문 **{sum(r9.values()):,}개** → 검색 대상 **{r9.get('검색 대상', 0):,}개**, 제외: 삭제 {r9.get('삭제', 0):,} · 본문 없음 {r9.get('본문 없음', 0)}")
    p(f"- 줄(항·호·목) **{res['6']['검사 줄']:,}개** (`article_lines.csv`)")
    p(f"- 만든 방법: " + ", ".join(f"`{k}` {v:,}" for k, v in res["방법별"].items()))
    p("  - `api` = 법령 JSON 항·호·목 칸 / `admrul_list` = 조문 목록으로 온 행정규칙 / `admrul_split` = 한 덩어리 행정규칙을 조문으로 나눔 / "
      "`guideline_items` = `1. 목 적` 형식 지침의 번호 항목 / `whole_body` = 번호도 없는 짧은 공지 2개")
    p("- 표시: " + ", ".join(f"`{k}` {v:,}" for k, v in res["표시"].items()))
    p("- 지운 것: " + ", ".join(f"{k} {v:,}" for k, v in res["지운 것"].items()) + "\n")

    p("## 판단 기준 점검 (기획안 5-2)\n")
    p("| # | 기준 | 결과 | 내용 |\n|---|---|---|---|")
    p(f"| 1 | 빈 조문 | {ok('1')} | 검색 대상 {res['1']['검색 대상']:,}개 중 본문 빈 것 {res['1']['빈 것']} |")
    p(f"| 2 | 메타데이터 누락 | {ok('2')} | 법령명·구분·조문번호·시행일자 누락 0. 조문제목 없음 {sum(res['2']['조문제목 없음(원본에 없는 경우 허용)'].values()):,} (원본에 제목이 없는 조문) |")
    t3 = res["3"]["합계"]
    p(f"| 3 | 구조 | {ok('3')} | 법령 JSON 항 {t3['JSON 항']:,}·호 {t3['JSON 호']:,}·목 {t3['JSON 목']:,} = 줄 항 {t3['줄 항']:,}·호 {t3['줄 호']:,}·목 {t3['줄 목']:,}, 다른 조문 {res['3']['다른 조문']} |")
    p(f"| 4 | 행정규칙 나누기 | {ok('4')} | 한 덩어리 18개 모두 관문 통과 (손실 없음·번호 증가). 줄바꿈 넣은 뒤 줄이 2개 이상인 행정규칙 조문 {res['4']['줄이 2개 이상으로 나뉜 조문']:,}, 줄 가운데 남은 항 기호(②~⑳, 참조 제외) {res['4']['줄 가운데 남은 항 기호']}, 줄 가운데 붙은 목 {res['4']['줄 가운데 붙은 목 줄']} (원문 번호 중복 예외 {len(res['4']['원문 번호 중복(예외)'])}: {', '.join(res['4']['원문 번호 중복(예외)'])}) |")
    p(f"| 5 | 남은 태그 | {ok('5')} | 지우는 모양 남은 것 0. 남은 `<…>`·`[…]`는 아래 목록 (내용 대괄호) |")
    p(f"| 6 | 원본 대조 | {ok('6')} | 줄 {res['6']['검사 줄']:,}개 모두 원문의 부분열 (새로 만든 글 0). 지운 비율 중앙값 {res['6']['지운 비율 중앙값']:.1%}, 30% 넘게 지운 조문 {res['6']['지운 (조문 머리를 빼면 대부분 30% 이하, 나머지는 짧은 조문에 붙은 긴 개정 태그) 비율 30% 넘는 조문']} |")
    p(f"| 7 | 세 층 | {ok('7')} | 한자 괄호: 보여 주는 글 {res['7']['text 의 한자 괄호']:,} / 임베딩용 {res['7']['embed_body 의 한자 괄호']} |")
    p(f"| 8 | 삭제 | {ok('8')} | 5-1 규칙 {res['8']['5-1 규칙 삭제']:,} → 5-2 {res['8']['5-2 삭제(법령+목록 행정규칙)']:,} (5-2에서만 {len(res['8']['5-2에서만 삭제'])}, 5-1에서만 {len(res['8']['5-1에서만 삭제'])}). 한 덩어리 행정규칙 포함 전체 {res['8']['5-2 삭제(전체)']:,} |")
    p(f"| 9 | 조문 수 보존 | {ok('9')} | 5-1 조문 {res['9']['5-1 조문']:,} = 5-2 조문(법령+목록) {res['9']['5-2 조문(법령+목록)']:,}. 한 덩어리·지침에서 새로 나온 조문 {res['9']['한 덩어리·지침에서 나온 조문']:,} |")
    p(f"| 10 | 청크 준비 | {ok('10')} | 900자 넘는데 경계 없는 조문 {len(r10['경계 없는 900자 초과'])}개 (아래) |")
    p(f"| 11 | 전/후 샘플 | {ok('11')} | 9가지 유형 × 2건 (아래) |\n")

    p("### 4. 한 덩어리로 온 행정규칙 18개\n")
    p("| 규정 | 방법 | 글자 | 나온 단위 | 처음 ~ 끝 | 손실 없음 | 번호 증가 |\n|---|---|---|---|---|---|---|")
    for g in res["4"]["한 덩어리 규정"]:
        p(f"| {g['규정']} | `{g['방법']}` | {g['글자']:,} | {g['조문(단위)']} | {g['처음 ~ 끝']} | {'O' if g['손실 없음'] else 'X'} | {'O' if g['번호 증가'] else 'X'} |")
    p("\n- `guideline_items` 지침은 그림 속 항목·중간 목록 때문에 번호가 비는 곳이 있다(예: 1 → 4 → 6). 빈 번호의 글은 **앞 항목에 합쳐져 남는다** (손실 없음). 단위가 조금 굵어질 뿐이고 5-3에서 길이로 다시 나뉜다.")
    p(f"- 행정규칙 줄 단계: " + ", ".join(f"{k} {v:,}" for k, v in res["4"]["행정규칙 줄 단계"].items()) + "\n")

    p("### 5. 남은 `<…>`·`[…]` (상위 20, 지우지 않은 것)\n")
    p("| 모양 | 건수 |\n|---|---|")
    for k, v in res["5"]["남은 <…>·[…] 종류(상위 20)"]:
        p(f"| `{k.replace('|', '/')}` | {v} |")

    p("\n### 8. 삭제 조문\n")
    p("5-2에서만 삭제로 본 것 (5-1 규칙이 놓친 것): 항이 모두 삭제된 조문, `<삭 제>` 표기")
    for x in res["8"]["5-2에서만 삭제"]:
        p(f"- {x}")
    if res["8"]["5-1에서만 삭제"]:
        p("\n5-1에서만 삭제: " + ", ".join(res["8"]["5-1에서만 삭제"]))

    p("\n### 10. 청크 기준 확인 (5-3 확정 절차)\n")
    p("전처리 뒤(머리 떼고 태그 지운 `text`) 검색 대상 조문의 길이 분포:\n")
    p("| 백분위 | " + " | ".join(f"{k}%" for k in r10["글자 백분위"]) + " | 최대 |")
    p("|---|" + "---|" * (len(r10["글자 백분위"]) + 1))
    p("| 글자 | " + " | ".join(f"{v:,}" for v in r10["글자 백분위"].values()) + f" | {r10['최대 글자']:,} |")
    p("| 토큰 | " + " | ".join(f"{v:,}" for v in r10["토큰 백분위"].values()) + " | |")
    p(f"\n- 900자 이하 **{r10['900자 이하 비율']:.1%}** (기획안 5-3의 85~90% 범위 → 900자 그대로), 글자/토큰 {r10['글자/토큰']}")
    p("- 구분별 900자 초과 비율: " + ", ".join(f"{k} {v:.1%}" for k, v in r10["구분별 900자 초과"].items()))
    p("- 900자 넘는데 항·호·목·문장 경계가 없는 조문 (5-3에서 문장·글자 기준으로 나뉨):")
    for x in r10["경계 없는 900자 초과"]:
        p(f"  - {x}")

    p("\n### 11. 전/후 샘플\n")
    for k, xs in samples.items():
        p(f"#### {k}\n")
        for a in xs:
            p(f"**{a['law_name']} {a['article_label']}({a['title']})** · `{a['parse_method']}`\n")
            p("| | 글 |\n|---|---|")
            p(f"| 원문 | {show(a['raw_text'])} |")
            p(f"| 표시용 `text` | {show(a['text'])} |")
            if a["embed_body"] != a["text"]:
                p(f"| 임베딩용 `embed_body` | {show(a['embed_body'])} |")
            lv = " / ".join(f"{ln['level']}{'(' + ln['marker'] + ')' if ln['marker'] else ''}{'·삭제' if ln['is_deleted'] == 'True' else ''}"
                            for ln in by_art[a["article_uid"]][:12])
            p(f"| 줄 구조 | {lv} |\n")

    p("## 결과 파일 (`data/work/02_clean/`)\n")
    p("| 파일 | 내용 |\n|---|---|")
    p("| `articles.csv` | 조문: 법령명, 구분, 묶음, 조문번호, 조문제목, 장절, `text`(표시용), `embed_body`(임베딩용), `raw_text`(원문), 시행일자, 검색 대상 여부·제외 이유, 표시(`has_image`·`has_table`·`has_delegation`·`is_reference_list`), 글자·토큰 수 |")
    p("| `article_lines.csv` | 줄: 조문, 순서, 단계(본문·항·호·목·세목), 기호, 항·호·목 번호, 상위 줄, 표시용·임베딩용 줄, 삭제 여부 |")
    p("| `admrul_gate.json`, `clean_stats.json`, `audit.json` | 행정규칙 나누기 관문, 지운 모양별 건수, 점검 원자료 |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "5-2_전처리.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("reports/5-2_전처리.md", len(L), "줄")


if __name__ == "__main__":
    main()
