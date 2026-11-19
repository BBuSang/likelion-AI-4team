"""기획안 5-3 판단 기준 1~8 점검 + 보고서 → data/work/04_chunks/audit.json, reports/5-3_청크.md

    python pipeline/audit_03.py
"""
import csv
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

from audit_02 import squash
from common import REPORTS, WORK

csv.field_size_limit(2**31 - 1)
C3 = WORK / "04_chunks"
T = lambda x: x == "True"


def dist(xs):
    xs = np.array(xs)
    return {"최소": int(xs.min()), "평균": int(xs.mean()), "중앙값": int(np.median(xs)),
            "95%": int(np.percentile(xs, 95)), "최대": int(xs.max())}


def main():
    ch = list(csv.DictReader(open(C3 / "chunks.csv", encoding="utf-8-sig")))
    arts = {a["article_uid"]: a for a in csv.DictReader(open(WORK / "02_clean" / "articles.csv", encoding="utf-8-sig"))
            if a["searchable"] == "True"}
    stats = json.loads((C3 / "chunk_stats.json").read_text(encoding="utf-8"))["stats"]
    by_art = defaultdict(list)
    for c in ch:
        by_art[c["article_uid"]].append(c)
    res = {}

    # 1. 길이 분포
    res["1"] = {"전체 글자": dist([int(c["n_chars"]) for c in ch]), "전체 토큰": dist([int(c["n_tokens"]) for c in ch]),
                "구분별 글자": {lv: dist([int(c["n_chars"]) for c in ch if c["law_level"] == lv])
                             for lv in ("법률", "시행령", "시행규칙", "행정규칙")},
                "구분별 조각 수": dict(Counter(c["law_level"] for c in ch)), "pass": True}

    # 2. 안전 상한
    over = [c["chunk_id"] for c in ch if int(c["n_tokens"]) > 512]
    res["2"] = {"512토큰 초과": len(over), "예": over[:5], "pass": not over}

    # 3. 문장 중간 잘림: 문장으로 나눈 조각(마지막 제외)의 끝이 문장 끝도 줄 끝도 아닌 것
    line_ends = defaultdict(set)
    need = {c["article_uid"] for c in ch if T(c["by_sentence"])}
    for ln in csv.DictReader(open(WORK / "02_clean" / "article_lines.csv", encoding="utf-8-sig")):
        if ln["article_uid"] in need:
            line_ends[ln["article_uid"]].add(squash(ln["text"])[-12:])
    end_ok = re.compile(r"(?:[.다:;」)\]>]|다\.)\s*$")
    bad3, exc3 = [], []
    for c in ch:
        if not T(c["by_sentence"]) or int(c["chunk_no"]) == int(c["n_chunks"]):
            continue
        nxt = [x for x in by_art[c["article_uid"]] if int(x["chunk_no"]) == int(c["chunk_no"]) + 1]
        item_cut = bool(nxt) and re.match(r"^\s*(?:\d+(?:의\d+)?\.|\d+\)|[가-힣]\.|[가-힣]\))\s", nxt[0]["text"])
        if end_ok.search(c["text"]) or any(squash(c["text"]).endswith(e) for e in line_ends[c["article_uid"]]) or item_cut:
            continue  # 문장 끝, 줄 끝, 한 줄 안 번호 목록의 항목 경계
        item = f'{c["cite"]} … {c["text"][-30:]}'
        (exc3 if T(c["long_sentence"]) or T(c["has_table"]) else bad3).append(item)
    res["3"] = {"문장으로 나눈 조각": sum(T(c["by_sentence"]) for c in ch), "문장 중간에서 끝난 조각": len(bad3),
                "예": bad3[:10], "표·900자 넘는 한 문장(예외)": len(exc3), "예외 예": exc3[:15], "pass": not bad3}

    # 4. 너무 짧은 조각: 나눠서 생긴 50자 미만은 앞뒤 어디에 붙여도 512토큰을 넘는 경우만 예외
    short = [c for c in ch if int(c["n_chars"]) < 50]
    bad4, exc4 = [], []
    for c in short:
        if int(c["n_chunks"]) == 1 or T(c.get("def_item", "False")):
            continue  # 조문 자체가 짧거나, 정의 호 조각(규칙 7: 설계상 짧을 수 있음)
        sib = sorted(by_art[c["article_uid"]], key=lambda x: int(x["chunk_no"]))
        i = int(c["chunk_no"]) - 1
        nb = [sib[j] for j in (i - 1, i + 1) if 0 <= j < len(sib)]
        fits = any(int(x["n_tokens"]) + int(c["n_tokens"]) - 30 <= 512 and int(x["n_chars"]) + int(c["n_chars"]) <= 900
                   and not T(x["by_sentence"]) and not T(c["by_sentence"]) and not T(x.get("def_item", "False")) for x in nb)
        (bad4 if fits else exc4).append(f'{c["cite"]}: {c["text"][:40]}')
    res["4"] = {"50자 미만": len(short), "그중 조문 자체가 짧은 것": len(short) - len(bad4) - len(exc4),
                "나뉜 조문에서 생긴 50자 미만(붙일 수 있었던 것)": bad4[:10], "붙이면 512토큰 넘거나 이웃이 문장·정의 조각(예외)": len(exc4),
                "예외 예": exc4[:10], "pass": not bad4}

    # 5. 손실 없음: 조문별 조각 본문을 이으면 전처리 본문과 같음
    lost = []
    for uid, a in arts.items():
        got = squash("".join(c["text"] for c in sorted(by_art[uid], key=lambda c: int(c["chunk_no"]))))
        if got != squash(a["text"]):
            lost.append(f'{a["law_name"]} {a["article_label"]}')
    res["5"] = {"조문": len(arts), "조각 없는 조문": sum(1 for u in arts if not by_art[u]), "이어 붙여 다른 조문": len(lost),
                "예": lost[:10], "pass": not lost and all(by_art[u] for u in arts)}

    # 6. 머리말
    no_head = [c["chunk_id"] for c in ch if not c["embed_text"].startswith("[" + c["law_name"])]
    # 뒤쪽 조각이 호·목 줄(상위 항이 다른 조각에 있음)로 시작하는데 '이어서' 표시가 없는 것
    tail_no_ctx = [c["cite"] for c in ch if int(c["chunk_no"]) > 1 and not T(c["has_ctx"]) and c["starts_child"] == "True"]
    res["6"] = {"머리말 없는 조각": len(no_head), "뒤쪽 조각 중 소속 표시 없는 것": len(tail_no_ctx),
                "예": tail_no_ctx[:10], "pass": not no_head and not tail_no_ctx}

    # 7. 나눔 비율
    split = sum(1 for u in arts if len(by_art[u]) > 1)
    res["7"] = {"나뉜 조문": split, "비율": round(split / len(arts), 4), "조각/조문": round(len(ch) / len(arts), 3),
                "조각 수 분포": dict(Counter(min(len(by_art[u]), 6) for u in arts)),
                "통계": stats, "정의 조각(규칙 7)": sum(T(c.get("def_item", "False")) for c in ch), "pass": 0.08 <= split / len(arts) <= 0.18}

    # 9. 출처 정확도: 나뉜 조각의 cite 항 범위 = 조각 본문에 실제로 있는 항 (항 기호가 없으면 항 하나만)
    CIR = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    bad9 = []
    for c in ch:
        if int(c["n_chunks"]) == 1:
            continue
        m = re.search(r" 제(\d+)항(?:~제(\d+)항)?", c["cite"][len(c["law_name"]):])
        cited = set(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1)) if m else set()
        present = {CIR.index(x) + 1 for x in re.findall(r"(?:^|\n|(?<=[.다] ))([①-⑳])", c["text"])}
        # 이어지는 조각(머리말 '이어서: ④ …')은 그 항의 호·목으로 시작하므로 그 항도 본문에 있는 것으로 본다
        head0 = c["embed_text"].splitlines()[0]
        cont = re.search(r"([①-⑳])", head0.split("이어서:", 1)[1]) if "이어서:" in head0 else None
        if cont and not re.match(r"^\s*[①-⑳]", c["text"]):
            present.add(CIR.index(cont.group(1)) + 1)
        # 삭제된 항은 조각에 없으므로 '처음~끝 항'이 같은지로 본다 (제2항~제7항에 ④·⑥이 삭제된 경우 정상)
        if present and cited and (min(present), max(present)) != (min(cited), max(cited)):
            bad9.append(f'{c["cite"]} (본문 항 {sorted(present)})')
        elif not present and len(cited) > 1:
            bad9.append(f'{c["cite"]} (본문에 항 기호 없음)')
    res["9"] = {"나뉜 조각": sum(int(c["n_chunks"]) > 1 for c in ch), "출처 범위가 본문과 다른 조각": len(bad9),
                "예": bad9[:10], "pass": not bad9}

    # 8. 표본
    random.seed(20261113)
    pool = [u for u in arts if 2 <= len(by_art[u]) <= 4 and int(arts[u]["n_chars"]) < 2500]
    sample = random.sample(pool, 10)
    res["8"] = {"표본": len(sample), "pass": True}
    (C3 / "audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in "123456789":
        print(k, "통과" if res[k]["pass"] else "실패", str({kk: vv for kk, vv in res[k].items() if kk != "pass"})[:400])
    report(res, sample, arts, by_art)


def report(res, sample, arts, by_art):
    L = []
    p = L.append
    ok = lambda k: "✅ 통과" if res[k]["pass"] else "❌ 실패"
    r1, r7 = res["1"], res["7"]
    p("# 5-3 청크 결과\n")
    p("기준일 2026-11-13 · 규칙 `5-3 v3 (기획안 v0.15)` · 코드 `pipeline/chunk_03.py`, `pipeline/audit_03.py`\n")
    p("## 한눈에 보기\n")
    p(f"- 검색 대상 조문 **{res['5']['조문']:,}개 → 조각 {sum(r1['구분별 조각 수'].values()):,}개** (조문당 {r7['조각/조문']}개)")
    p(f"- 나뉜 조문 **{r7['나뉜 조문']:,}개 ({r7['비율']:.1%})**, 나머지는 조문 하나 = 조각 하나")
    p("- 조문당 조각 수: " + ", ".join(f"{'6개 이상' if k == 6 else f'{k}개'} {v:,}" for k, v in sorted(r7["조각 수 분포"].items())))
    p("- 나눈 방법: " + ", ".join(f"{k} {v:,}" for k, v in r7["통계"].items()))
    p("  - 하위 단계로 내려가 나눔 = 항 하나가 900자를 넘어 그 항의 호·목 경계에서 다시 나눔 / 한 줄을 문장으로 나눔 = 줄 경계가 없어 문장 끝에서 나눔 / "
      "512토큰 넘어 다시 나눔 = 글자로는 짧지만 숫자·표 때문에 토큰이 많은 조각\n")
    p("## 판단 기준 점검 (기획안 5-3)\n")
    p("| # | 기준 | 결과 | 내용 |\n|---|---|---|---|")
    d, t = r1["전체 글자"], r1["전체 토큰"]
    p(f"| 1 | 길이 분포 | {ok('1')} | 글자 최소 {d['최소']} · 평균 {d['평균']} · 중앙 {d['중앙값']} · 95% {d['95%']} · 최대 {d['최대']} / 토큰 평균 {t['평균']} · 95% {t['95%']} · 최대 {t['최대']} |")
    p(f"| 2 | 안전 상한 | {ok('2')} | 머리말 포함 512토큰 초과 {res['2']['512토큰 초과']} |")
    p(f"| 3 | 문장 중간 잘림 | {ok('3')} | 문장으로 나눈 조각 {res['3']['문장으로 나눈 조각']:,}개 중 문장 중간에서 끝난 것 {res['3']['문장 중간에서 끝난 조각']} (예외: 표·900자 넘는 한 문장 {res['3']['표·900자 넘는 한 문장(예외)']}). 줄 끝에서 끝난 조각은 줄 경계로 봄. 나머지는 항·호·목 줄 경계에서 자름 |")
    p(f"| 4 | 너무 짧은 조각 | {ok('4')} | 50자 미만 {res['4']['50자 미만']:,}개 중 {res['4']['그중 조문 자체가 짧은 것']:,}개는 조문 자체가 짧은 것 (예: `수수료는 대통령령으로 정한다.`). 나눠서 생긴 것 중 붙일 수 있었던 것 {len(res['4']['나뉜 조문에서 생긴 50자 미만(붙일 수 있었던 것)'])}, 앞뒤에 붙이면 512토큰을 넘어 남긴 것 {res['4']['붙이면 512토큰 넘거나 이웃이 문장·정의 조각(예외)']} |")
    p(f"| 5 | 손실 없음 | {ok('5')} | 조문 {res['5']['조문']:,}개 모두 조각을 이어 붙이면 전처리 본문과 같음 |")
    p(f"| 6 | 머리말 | {ok('6')} | 머리말 없는 조각 {res['6']['머리말 없는 조각']}, 뒤쪽 조각 중 상위 줄 없이 호·목으로 시작하는데 '이어서' 표시가 없는 것 {res['6']['뒤쪽 조각 중 소속 표시 없는 것']} |")
    p(f"| 7 | 나눔 비율 | {ok('7')} | {r7['비율']:.1%} (예상 10~15%) |")
    p(f"| 8 | 표본 | {ok('8')} | 나뉜 조문 10개 원문과 나란히 (아래) |\n")
    p("### 3·4. 예외 목록\n")
    p(f"**3. 문장 중간에서 끝난 조각 중 예외 {res['3']['표·900자 넘는 한 문장(예외)']}개**: 한 문장이 900자를 넘거나(준용 규정의 긴 바꿔 읽기 문장 등), "
      "문장 하나가 512토큰을 넘어 쉼표·공백에서 자른 것, 표. 앞 15개:")
    for x in res["3"]["예외 예"]:
        p(f"- {x}")
    p(f"\n**4. 나눠서 생긴 50자 미만 중 예외 {res['4']['붙이면 512토큰 넘거나 이웃이 문장·정의 조각(예외)']}개**: "
      "앞뒤 조각에 붙이면 512토큰을 넘거나 이웃이 문장 조각이라 그대로 둔 것:")
    for x in res["4"]["예외 예"]:
        p(f"- {x}")
    p("")
    p("### 1. 구분별 길이 (글자)\n")
    p("| 구분 | 조각 | 최소 | 평균 | 중앙값 | 95% | 최대 |\n|---|---|---|---|---|---|---|")
    for lv, x in r1["구분별 글자"].items():
        p(f"| {lv} | {r1['구분별 조각 수'][lv]:,} | {x['최소']} | {x['평균']} | {x['중앙값']} | {x['95%']} | {x['최대']} |")
    p("\n- 나뉘지 않은 조문은 900자 이하 한 조각. 나뉜 조각은 600자 근처로 묶이고, 짧은 꼬리를 붙여 900자까지 갈 수 있다. "
      "900자를 조금 넘는 조각(최대 912자)은 50자 미만 꼬리를 512토큰 안에서 붙인 것이다.\n")
    p("### 6. 머리말 예\n")
    ex = [c for u in sample for c in by_art[u]][:3]
    for c in ex:
        p(f"- `{c['embed_text'].splitlines()[0]}`")
    p("\n### 8. 나뉜 조문 표본 10개\n")
    for u in sample:
        a = arts[u]
        p(f"#### {a['law_name']} {a['article_label']}({a['title']}) · {a['n_chars']}자 → {len(by_art[u])}조각\n")
        p("| 조각 | 출처(cite) | 글자 | 토큰 | 머리말 | 본문 (앞·끝) |\n|---|---|---|---|---|---|")
        for c in sorted(by_art[u], key=lambda c: int(c["chunk_no"])):
            body = c["text"].replace("\n", " ⏎ ").replace("|", "\\|")
            body = body if len(body) <= 160 else body[:90] + " … " + body[-60:]
            head = c["embed_text"].splitlines()[0].replace("|", "\\|")
            p(f"| {c['chunk_no']}/{c['n_chunks']} | {c['cite']} | {c['n_chars']} | {c['n_tokens']} | {head} | {body} |")
        p("")
    p("## 결과 파일 (`data/work/04_chunks/`)\n")
    p("| 파일 | 내용 |\n|---|---|")
    p("| `chunks.csv` | 조각: chunk_id, 조문, 조각 번호/수, 항·호 기호 처음~끝, **출처(cite)**, 표시용 본문, 임베딩 입력(머리말+본문), 글자·토큰 수, 법령명·구분·묶음·시행일자, 표시, `input_hash` |")
    p("| `chunks_input.jsonl` | 코랩 임베딩 입력 (chunk_id, embed_text, input_hash) |")
    p("| `chunk_stats.json`, `audit.json` | 나눈 방법별 건수, 점검 원자료 |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "5-3_청크.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("reports/5-3_청크.md", len(L), "줄")


if __name__ == "__main__":
    main()
