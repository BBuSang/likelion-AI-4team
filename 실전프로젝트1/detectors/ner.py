"""토큰 분류 모델 공통 래퍼 (Hugging Face transformers, CPU).

- 긴 문서는 510토큰 창을 128토큰씩 겹쳐 밀고, 겹친 구간은 창 가운데 쪽 예측만 씀
- BIO/BIOES 태그를 글자 구간으로 복원 (토크나이저 offset mapping 사용, 원문 위치 그대로)
- 모델 라벨을 우리 7종으로 변환하고, 대응 없는 라벨(기관·부서·직함·날짜 등)은 버림
- 모델 저장소의 추가 코드(veil.py 등)는 실행하지 않고, 가중치·토크나이저만 내려받아 씀
"""
MODELS = {
    # 1T/veil-pii-ko-lite: KoELECTRA-base, BIOES 32종
    "veil": ("1T/veil-pii-ko-lite", {
        "PERSON": "person", "PHONE": "phone_number", "EMAIL": "email", "ADDRESS": "address",
        "RRN": "resident_registration_number", "ACCOUNT_NUMBER": "account_number",
        "CARD_NUMBER": "card_number", "VIRTUAL_CARD_NUMBER": "card_number"}),
    # seongyeon1/ko-pii-ner-roberta-base: klue/roberta-base, BIO 20종
    "seongyeon": ("seongyeon1/ko-pii-ner-roberta-base", {
        "PERSON_NAME": "person", "PHONE_NUM": "phone_number", "EMAIL": "email", "ADDRESS": "address",
        "RRN": "resident_registration_number", "ACCOUNT_NUM": "account_number", "CARD_NUM": "card_number"}),
    # townboy/kpfbert-kdpii: KPF-bert-ner(113M), BIO 33종, KDPII 학습. 휴대전화·일반전화를 phone으로 묶음
    "kpf": ("townboy/kpfbert-kdpii", {
        "PS_NAME": "person", "QT_MOBILE": "phone_number", "QT_PHONE": "phone_number", "TMI_EMAIL": "email",
        "LC_ADDRESS": "address", "QT_RESIDENT_NUMBER": "resident_registration_number",
        "QT_ACCOUNT_NUMBER": "account_number", "QT_CARD_NUMBER": "card_number"}),
    # atonlee/koelectra-ko-pii-ner: KoELECTRA-small(14M), BIO 29종
    "atonlee": ("atonlee/koelectra-ko-pii-ner", {
        "NAME": "person", "PHONE": "phone_number", "EMAIL": "email", "ADDRESS": "address",
        "RRN": "resident_registration_number", "ACCOUNT_NUMBER": "account_number", "CARD_NUMBER": "card_number"}),
    # YakuzaNeko/kr-dlp-ner-roberta-large: klue/roberta-large(336M), BIO 14종. BANK = 계좌번호
    "dlp": ("YakuzaNeko/kr-dlp-ner-roberta-large", {
        "PER": "person", "PHONE": "phone_number", "EMAIL": "email", "ADDR": "address",
        "RRN": "resident_registration_number", "BANK": "account_number", "CARD": "card_number"}),
    # FrameByFrame/korean-pii-e5-base: multilingual-e5-base(278M), BIOES 9종.
    # account_number 하나에 계좌·카드·주민번호가 묶여 있어 account_number로만 대응 (카드·주민번호는 라벨 오류로 나타남)
    "e5": ("FrameByFrame/korean-pii-e5-base", {
        "private_person": "person", "private_phone": "phone_number", "private_email": "email",
        "private_address": "address", "account_number": "account_number"}),
}

WINDOW, STRIDE = 510, 128


class TokenClassifier:
    def __init__(self, name):
        self.model_id, self.label_map = MODELS[name]
        self.tok = self.model = None

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            # CPU 기준: bfloat16으로 저장된 모델(e5)도 float32로 올림
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_id, dtype=torch.float32).eval()

    def tags(self, text):
        """토큰별 (태그, (start, end)) 목록. 창을 나눠 돌리고 겹친 구간은 가운데 쪽 예측을 씀."""
        import torch
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        id2label = self.model.config.id2label
        preds = [None] * len(ids)
        starts = list(range(0, max(len(ids) - STRIDE, 1), WINDOW - STRIDE))
        for k, s in enumerate(starts):
            e = min(s + WINDOW, len(ids))
            window = [self.tok.cls_token_id] + ids[s:e] + [self.tok.sep_token_id]
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([window])).logits[0, 1:-1]
            keep_from = s + (STRIDE // 2 if k > 0 else 0)
            keep_to = e - (STRIDE // 2 if k < len(starts) - 1 else 0)
            for i, p in enumerate(logits.argmax(-1).tolist()):
                if keep_from <= s + i < keep_to:
                    preds[s + i] = id2label[p]
        return list(zip(preds, offsets))

    def predict(self, text):
        """BIO/BIOES 태그를 구간으로 묶음. I/E가 다른 라벨 뒤에 오면 새 구간으로 시작(관대한 복원)."""
        self.load()
        spans, cur = [], None
        for tag, (s, e) in self.tags(text):
            prefix, _, typ = (tag or "O").partition("-")
            if prefix == "O":
                cur = None
                continue
            if prefix in ("B", "S") or cur is None or cur["type"] != typ:
                cur = {"type": typ, "start": s, "end": e}
                spans.append(cur)
            else:
                cur["end"] = e
            if prefix in ("E", "S"):
                cur = None
        return [{"label": self.label_map[sp["type"]], "start": sp["start"], "end": sp["end"]}
                for sp in spans if sp["type"] in self.label_map]
