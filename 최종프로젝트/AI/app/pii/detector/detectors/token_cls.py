"""방식 2: 토큰 분류(NER) 모델 탐지기 (Hugging Face).

모델마다 라벨 체계가 달라 우리 라벨 7종으로 바꾸는 LABEL_MAP 이 핵심이다.
매핑되지 않는 라벨(ORGANIZATION, DATE, POSITION 등)은 버린다.
"""
from transformers import pipeline

from postprocess import drop_decoys, drop_venue_addresses, fix_boundaries

MODELS = {
    # 별칭: (Hugging Face 모델 ID, 모델 라벨 → 우리 라벨)
    "koelectra_pii": ("atonlee/koelectra-ko-pii-ner", {
        "NAME": "person", "PHONE": "phone_number", "EMAIL": "email", "ADDRESS": "address",
        "RRN": "resident_registration_number", "ACCOUNT_NUMBER": "account_number",
        "CARD_NUMBER": "card_number",
    }),
    # 카드번호 라벨 없음
    "piilot": ("ParkJunSeong/PIILOT_NER_Model", {
        "p_nm": "person", "p_ph": "phone_number", "p_em": "email", "p_add": "address",
        "p_rrn": "resident_registration_number", "p_acn": "account_number",
    }),
}


# 서브워드 조각을 개체로 묶는 방식. simple 은 조각 단위라 PIILOT 처럼 '김은'+'영'으로 쪼개지는 모델이 있고,
# first 는 띄어쓰기 단어 단위라 쪼개짐은 없지만 '윤선영은'처럼 조사까지 붙을 수 있다 → dev 에서 모델별로 골라 씀
AGGREGATION = {"koelectra_pii": "simple", "piilot": "first"}


class TokenClsDetector:
    def __init__(self, alias: str, aggregation: str | None = None, postprocess: bool = True):
        model_id, self.label_map = MODELS[alias]
        self.name = alias
        self.postprocess = postprocess  # 경계 보정 + 미끼 오탐 제거 (postprocess.py)
        # device=-1: 운영 환경(GPU 없는 CPU 서버) 기준으로 속도를 잰다
        self.pipe = pipeline("token-classification", model=model_id, device=-1,
                             aggregation_strategy=aggregation or AGGREGATION[alias])

    def detect(self, text: str) -> list:
        # 줄 단위로 넣는다: 최대 길이(512토큰) 초과를 막고, 줄을 넘어 두 개체가 합쳐지는 것도 막는다
        lines, offsets, pos = [], [], 0
        for line in text.split("\n"):
            if line.strip():
                lines.append(line)
                offsets.append(pos)
            pos += len(line) + 1
        ents = []
        if not lines:
            return ents
        for off, found in zip(offsets, self.pipe(lines)):
            for ent in found:
                label = self.label_map.get(ent["entity_group"])
                if label:
                    ents.append({"label": label, "start": off + ent["start"], "end": off + ent["end"],
                                 "confidence": round(float(ent["score"]), 4)})
        if self.postprocess:
            ents = drop_venue_addresses(text, drop_decoys(text, fix_boundaries(text, ents)))
        return ents


class HFNerDetector:
    """팀원 래퍼(detectors/hf_ner.py)로 돌리는 토큰 분류 모델: veil, seongyeon, dlp, kpf, e5, atonlee.
    pipeline 의 묶는 방식(aggregation) 대신 BIO/BIOES 태그를 직접 글자 구간으로 복원하고, 긴 문서는 창을 나눠 처리한다.
    비교를 공정하게 하려고 우리 후처리(경계 보정·대표번호·장소 주소 제외)를 똑같이 붙인다."""

    def __init__(self, name: str, postprocess: bool = True, line_mode: str = "first"):
        """line_mode: 줄바꿈을 넘어 이어진 개체 처리
        split_all = 줄에서 잘라 모든 조각 유지 / first = 첫 조각만 / per_line = 줄마다 따로 모델에 넣기"""
        from detectors.hf_ner import TokenClassifier
        self.name = name
        self.line_mode = line_mode
        self.model = TokenClassifier(name)
        self.model.load()
        self.postprocess = postprocess

    def detect(self, text: str) -> list:
        ents = []
        if self.line_mode == "per_line":
            # 줄마다 따로 넣는다: 줄을 넘어 개체가 이어질 수 없음 (호출 수가 줄 수만큼 늘어남)
            pos = 0
            for line in text.split("\n"):
                if line.strip():
                    ents += [{**e, "start": pos + e["start"], "end": pos + e["end"]} for e in self.model.predict(line)]
                pos += len(line) + 1
        else:
            # 문서 전체를 한 번에 넣으면 '김예준\n을'처럼 줄바꿈을 넘어 개체가 이어지므로 줄에서 자른다
            for e in self.model.predict(text):
                start = e["start"]
                for k, part in enumerate(text[e["start"]:e["end"]].split("\n")):
                    if part.strip() and (k == 0 or self.line_mode == "split_all"):
                        ents.append({**e, "start": start, "end": start + len(part)})
                    start += len(part) + 1
        if self.postprocess:
            ents = drop_venue_addresses(text, drop_decoys(text, fix_boundaries(text, ents)))
        return ents
