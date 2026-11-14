"""적재 입력 읽기·구간 확인·검수 차이 계산 (DB·모델 필요 없음)."""
import json

import pytest

from app.ingest import service

TEXT = "고객명: 김민준\n연락처: 010-1234-5678\n주소: 서울시 강남구"


def test_parse_jsonl_keeps_gold_only_for_compare():
    line = json.dumps({"id": "H1", "split": "holdout", "doc_type": "상담", "text": TEXT, "channel": "전화",
                       "entities": [{"label": "person", "start": 5, "end": 8, "text": "김민준"},
                                    {"label": "organization", "start": 0, "end": 2}]}, ensure_ascii=False)
    [d] = service.parse_upload("jsonl", line + "\n\n")
    assert d["ext_id"] == "H1" and d["doc_type"] == "상담"
    assert d["gold"] == [{"label": "person", "start": 5, "end": 8}]  # 모르는 라벨은 뺌
    assert d["meta"] == {"channel": "전화"}  # split·entities 는 메타로 안 감


def test_parse_errors():
    with pytest.raises(service.IngestError):
        service.parse_upload("jsonl", '{"id": 1}')
    with pytest.raises(service.IngestError):
        service.parse_upload("jsonl", "not json")
    with pytest.raises(service.IngestError):
        service.parse_upload("text", "   ")
    with pytest.raises(service.IngestError):
        service.parse_upload("csv", "a,b")


def test_validate_spans():
    ok = service.validate_spans(TEXT, [{"label": "phone_number", "start": 13, "end": 26},
                                       {"label": "person", "start": 5, "end": 8, "text": "김민준"}])
    assert [s["start"] for s in ok] == [5, 13]  # 정렬
    bad = [
        [{"label": "person", "start": 5, "end": 8, "text": "김민"}],          # 글자 불일치
        [{"label": "nickname", "start": 5, "end": 8}],                       # 모르는 라벨
        [{"label": "person", "start": 5, "end": 999}],                       # 본문 밖
        [{"label": "person", "start": 4, "end": 5}],                         # 공백만
        [{"label": "person", "start": 5, "end": 8}, {"label": "person", "start": 7, "end": 9}],  # 겹침
    ]
    for spans in bad:
        with pytest.raises(service.IngestError):
            service.validate_spans(TEXT, spans)


def test_diff_counts_reviewer_changes():
    det = [{"label": "person", "start": 5, "end": 8}, {"label": "address", "start": 13, "end": 26},
           {"label": "email", "start": 30, "end": 35}]
    final = [{"label": "person", "start": 5, "end": 8}, {"label": "phone_number", "start": 13, "end": 26},
             {"label": "address", "start": 31, "end": 38}]
    d = service.diff(det, final)
    assert (d["added"], d["removed"], d["relabeled"]) == (1, 1, 1)
