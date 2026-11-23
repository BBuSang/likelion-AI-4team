"""공개 자료 보기 API (/internal/public-docs/*). 스프링이 관리자에게만 열어 줌. 읽기만 (고치기·지우기 없음).

출처마다 저장 방식이 달라 따로 읽음
  - product  금융상품 설명서·약관: data/public_rag/product/meta/*.csv + chunks/chunks.jsonl (서버에만 있음, 없으면 섹션 없이)
  - law      법령: DB lawrag.chunks (읽기 전용 law_reader) — 법령 단위 목록, 상세는 조문 목록
  - noaction 금융당국 회신(법령해석·비조치의견서): DB interp.cases (읽기 전용 interp_reader)
"""
import csv
import importlib
import json
import threading
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException

from app.rag_public.product.index import CHUNKS, META

# 패키지 __init__ 에 같은 이름(search) 함수가 있어 'from … import search' 로는 모듈이 아닌 함수가 옴 → 모듈을 이름으로 가져옴
law_search = importlib.import_module("app.rag_public.law.search")
noaction_search = importlib.import_module("app.rag_public.noaction.search")

router = APIRouter(prefix="/public-docs", tags=["public-docs"])

Source = Literal["product", "law", "noaction"]
MAX_LIMIT = 100
SECTION_CHARS = 800   # 상세 화면에 섹션·조문마다 보여 주는 글자 수
ANSWER_CHARS = 4000   # 회신 질의·회답·이유 각각 최대 글자 수

_lock = threading.Lock()
_product = None


def _page(limit, offset):
    return max(1, min(limit, MAX_LIMIT)), max(0, offset)


def _cut(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


# ------------------------------------------------------------------ 금융상품

def _csv(name):
    with open(META / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _product_data():
    """문서 목록·상품 연결·섹션을 처음 한 번 읽어 둠 (파일이라 바뀌면 서버 재시작으로 반영)"""
    global _product
    with _lock:
        if _product is None:
            docs = _csv("documents.csv")
            names = {r["product_id"]: r["name"] for r in _csv("products.csv")}
            links = {}
            for r in _csv("product_docs.csv"):
                links.setdefault(r["doc_id"], []).append({"product_id": r["product_id"], "name": names.get(r["product_id"], r["product_id"]), "role": r["role"]})
            sections = {}
            path = CHUNKS / "chunks.jsonl"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            c = json.loads(line)
                            sections.setdefault(c["doc_id"], []).append(c)
            _product = {"docs": docs, "links": links, "sections": sections}
        return _product


def _product_list(q, limit, offset):
    d = _product_data()
    rows = d["docs"]
    if q:
        rows = [r for r in rows if q in r["doc_id"] or q in r["title"]
                or any(q in p["name"] for p in d["links"].get(r["doc_id"], []))]
    items = [{"id": r["doc_id"], "title": r["title"], "kind": r.get("doc_type") or r.get("role"),
              "products": len(d["links"].get(r["doc_id"], [])), "sections": len(d["sections"].get(r["doc_id"], []))}
             for r in rows[offset:offset + limit]]
    return {"total": len(rows), "items": items}


def _product_detail(doc_id):
    d = _product_data()
    doc = next((r for r in d["docs"] if r["doc_id"] == doc_id), None)
    if doc is None:
        raise HTTPException(404, "문서 없음")
    secs = [{"label": c.get("section") or "", "page": c.get("page"), "text": _cut(c.get("content"), SECTION_CHARS)}
            for c in d["sections"].get(doc_id, [])]
    return {"id": doc_id, "title": doc["title"], "kind": doc.get("doc_type") or doc.get("role"), "file": doc.get("file"),
            "products": d["links"].get(doc_id, []), "sections": secs, "sections_available": bool(d["sections"])}


# ------------------------------------------------------------------ 법령

def _law_list(q, limit, offset):
    like = f"%{q}%" if q else None
    with law_search._lock:
        conn = law_search.db()
        total = conn.execute("SELECT count(DISTINCT law_uid) FROM lawrag.chunks WHERE %(q)s::text IS NULL OR law_name ILIKE %(q)s",
                             {"q": like}).fetchone()[0]
        rows = conn.execute("""SELECT law_uid, law_name, law_level, max(effective_date), count(DISTINCT article_uid)
                               FROM lawrag.chunks WHERE %(q)s::text IS NULL OR law_name ILIKE %(q)s
                               GROUP BY law_uid, law_name, law_level ORDER BY law_name LIMIT %(l)s OFFSET %(o)s""",
                            {"q": like, "l": limit, "o": offset}).fetchall()
    return {"total": total, "items": [{"id": str(u), "title": n, "kind": lvl, "date": str(ef) if ef else None, "articles": a}
                                      for u, n, lvl, ef, a in rows]}


def _law_detail(law_uid):
    with law_search._lock:
        rows = law_search.db().execute("""SELECT article_uid, article_label, title, law_name, law_level, effective_date, chunk_no, text
                                          FROM lawrag.chunks WHERE law_uid::text = %s ORDER BY article_uid, chunk_no""",
                                       (law_uid,)).fetchall()
    if not rows:
        raise HTTPException(404, "법령 없음")
    arts, order = {}, []
    for auid, label, title, lname, lvl, ef, _no, text in rows:
        a = arts.get(auid)
        if a is None:
            arts[auid] = a = {"label": label or "", "title": title or "", "parts": []}
            order.append(auid)
        a["parts"].append(text or "")
    _, _, _, lname, lvl, ef, _, _ = rows[0]
    secs = [{"label": f"{arts[u]['label']}" + (f"({arts[u]['title']})" if arts[u]["title"] else ""),
             "text": _cut("\n".join(arts[u]["parts"]), SECTION_CHARS)} for u in order]
    return {"id": law_uid, "title": lname, "kind": lvl, "date": str(ef) if ef else None, "sections": secs}


# ------------------------------------------------------------------ 금융당국 회신 (법령해석·비조치의견서)

def _noaction_list(q, limit, offset):
    like = f"%{q}%" if q else None
    where = "WHERE %(q)s::text IS NULL OR title ILIKE %(q)s OR serial_no ILIKE %(q)s OR case_key ILIKE %(q)s"
    with noaction_search.connect() as conn:
        total = conn.execute(f"SELECT count(*) AS n FROM interp.cases {where}", {"q": like}).fetchone()["n"]
        rows = conn.execute(f"""SELECT case_key, gubun, serial_no, title, coalesce(category_final, category) AS category, reply_date
                                FROM interp.cases {where}
                                ORDER BY reply_date DESC NULLS LAST, case_key LIMIT %(l)s OFFSET %(o)s""",
                            {"q": like, "l": limit, "o": offset}).fetchall()
    return {"total": total, "items": [{"id": r["case_key"], "title": r["title"] or r["serial_no"] or r["case_key"], "kind": r["gubun"],
                                       "serial_no": r["serial_no"], "category": r["category"],
                                       "date": str(r["reply_date"]) if r["reply_date"] else None} for r in rows]}


def _noaction_detail(case_key):
    with noaction_search.connect() as conn:
        r = conn.execute("""SELECT case_key, gubun, serial_no, title, reply_title, coalesce(category_final, category) AS category,
                                   reg_date, reply_date, status, answer_type, question, answer, reason, url
                            FROM interp.cases WHERE case_key = %s""", (case_key,)).fetchone()
    if r is None:
        raise HTTPException(404, "회신 없음")
    secs = [{"label": name, "text": _cut(r[key], ANSWER_CHARS)} for name, key in (("질의 요지", "question"), ("회답", "answer"), ("이유", "reason")) if r[key]]
    return {"id": r["case_key"], "title": r["title"] or r["reply_title"] or r["case_key"], "kind": r["gubun"], "serial_no": r["serial_no"],
            "category": r["category"], "date": str(r["reply_date"]) if r["reply_date"] else None,
            "reg_date": str(r["reg_date"]) if r["reg_date"] else None, "status": r["status"], "answer_type": r["answer_type"],
            "url": r["url"], "sections": secs}


# ------------------------------------------------------------------

LISTS = {"product": _product_list, "law": _law_list, "noaction": _noaction_list}
DETAILS = {"product": _product_detail, "law": _law_detail, "noaction": _noaction_detail}


def _call(fn, *args):
    try:
        return fn(*args)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(503, "자료 파일이 없어요")
    except Exception as e:  # DB 연결 실패 등 — 출처 하나가 안 돼도 다른 출처는 보이게 503 으로
        raise HTTPException(503, f"자료를 읽지 못했어요 ({type(e).__name__})")


@router.get("/{source}")
def list_docs(source: Source, q: Optional[str] = None, limit: int = 30, offset: int = 0):
    limit, offset = _page(limit, offset)
    return _call(LISTS[source], (q or "").strip() or None, limit, offset)


@router.get("/{source}/{doc_id}")
def doc_detail(source: Source, doc_id: str):
    return _call(DETAILS[source], doc_id)
