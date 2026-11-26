"""검수 때 본문 고치기: 구간 옮기기 · 화자 줄 찾기 (DB·모델 필요 없음)."""
from app.ingest import service


def _sp(label, start, end):
    return {"label": label, "start": start, "end": end}


def test_remap_moves_spans_outside_edit():
    old = "김민수 고객 010-1234-5678 확인"
    new = "김민수 고객님 연락처 010-1234-5678 확인"
    kept, dropped, changed = service.remap_spans(old, new, [_sp("person", 0, 3), _sp("phone_number", 7, 20)])
    assert dropped == 0
    assert [new[k["start"]:k["end"]] for k in kept] == ["김민수", "010-1234-5678"]
    assert changed and all(0 <= a <= b <= len(new) for a, b in changed)


def test_remap_drops_spans_touching_edit():
    old = "담당 김민수 대리"
    new = "담당 박민수 대리"           # 이름 안을 고침 → 그 구간은 버리고 다시 탐지로
    kept, dropped, _ = service.remap_spans(old, new, [_sp("person", 3, 6)])
    assert kept == [] and dropped == 1


def test_remap_speaker_rename_keeps_body_spans():
    old = "[00:00] 화자1: 성함이요?\n[00:03] 화자2: 김민수입니다"
    new = "[00:00] 상담원: 성함이요?\n[00:03] 고객: 김민수입니다"
    i = old.index("김민수")
    kept, dropped, _ = service.remap_spans(old, new, [_sp("person", i, i + 3)])
    assert dropped == 0 and new[kept[0]["start"]:kept[0]["end"]] == "김민수"


def test_speakers_in():
    t = "[00:00] 화자1: 안녕하세요\n[00:05] 화자2: 네\n[01:02] 화자1: 시작할게요\n[12:30] 김 과장: 좋아요\n그냥 줄"
    assert service.speakers_in(t) == ["화자1", "화자2", "김 과장"]
    assert service.speakers_in("녹음이 아닌 글") == []
