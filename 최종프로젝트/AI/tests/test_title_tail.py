"""모델이 이름 뒤 직함까지 잡은 구간 다듬기 ('수빈 사원' → '수빈'). 탐지 모델 없이 후처리 함수만."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "pii" / "detector"))
from postprocess import drop_decoys, fix_boundaries, strip_person_tail  # noqa: E402


def names(text, word):
    s = text.index(word)
    ents = [{"label": "person", "start": s, "end": s + len(word)}]
    return [text[e["start"]:e["end"]] for e in drop_decoys(text, fix_boundaries(text, ents))]


def test_spaced_and_glued_titles_are_stripped():
    assert strip_person_tail("수빈 사원") == "수빈"
    assert strip_person_tail("예린사원") == "예린"
    assert strip_person_tail("수빈 주임님") == "수빈"
    assert strip_person_tail("다은 차장") == "다은"


def test_surname_plus_title_is_left_alone():
    # 떼면 성 한 글자만 남는 것은 그대로 (drop_decoys 가 따로 뺌)
    assert strip_person_tail("김사원") == "김사원"
    assert strip_person_tail("김 사원") == "김 사원"


def test_model_span_with_title_becomes_name_only():
    assert names("팀장님, 수빈 사원이 휴가입니다.", "수빈 사원") == ["수빈"]
    assert names("예린사원이 처리함", "예린사원") == ["예린"]  # 뒤에 붙은 직함은 단어 중간이 아님
    assert names("팀장아라대리님에게도 전달", "아라") == ["아라"]


def test_cut_word_still_dropped():
    assert names("미르보안 담당", "미르") == []  # 뒤에 직함이 아닌 글자가 붙으면 여전히 단어 중간
