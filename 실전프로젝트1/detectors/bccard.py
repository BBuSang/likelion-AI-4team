"""모델 단독 탐지기: BCCard/MoAI-Privacy-Filter-INT8 (openai/privacy-filter 파인튜닝, INT8 ONNX, Apache-2.0).

- transformers로는 로드되지 않아 ONNX Runtime(CPU)으로 실행. 저장소의 가중치·토크나이저·설정만 내려받아 씀
- 모델 카드의 처리 순서를 따름: 토크나이저 → ONNX 그래프 → FP32 logits → 제약 BIOES Viterbi → 글자 구간 → 공백 경계 정리
  (공백 경계 정리는 base.postprocess가 맡음)
- Viterbi: O/E/S 다음에는 O·B·S만, B/I 다음에는 같은 라벨의 I·E만 허용. 시작은 O·B·S, 끝은 O·E·S.
  전이 가중치는 viterbi_calibration.json의 default 값(현재 모두 0.0)을 씀
- 한 번에 최대 1024토큰. 긴 문서는 창을 128토큰씩 겹쳐 밀고, 겹친 구간은 창 가운데 쪽 logits를 씀
- 라벨 29종 중 7종에 대응하는 것만 씀. FRN(외국인등록번호)은 주민번호, VIRTUAL_CARD_NUMBER는 카드번호로 묶음
"""
import json
from pathlib import Path

import numpy as np

MODEL_ID = "BCCard/MoAI-Privacy-Filter-INT8"
LABEL_MAP = {
    "PERSON": "person", "PHONE": "phone_number", "EMAIL": "email", "ADDRESS": "address",
    "RRN": "resident_registration_number", "FRN": "resident_registration_number",
    "ACCOUNT_NUMBER": "account_number", "CARD_NUMBER": "card_number", "VIRTUAL_CARD_NUMBER": "card_number"}
MAX_TOKENS, STRIDE = 1024, 128

_tok = _sess = _labels = _trans = _start = _end = None


def load():
    global _tok, _sess, _labels, _trans, _start, _end
    if _sess is not None:
        return
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    d = Path(snapshot_download(repo_id=MODEL_ID))
    _tok = AutoTokenizer.from_pretrained(d)
    _sess = ort.InferenceSession(str(d / "model_quantized.onnx"), providers=["CPUExecutionProvider"])
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    _labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    bias = json.loads((d / "viterbi_calibration.json").read_text(encoding="utf-8"))
    _trans, _start, _end = transitions(_labels, bias["operating_points"]["default"]["biases"])


def transitions(labels, b):
    """허용 전이 행렬(불가는 -inf)과 시작·끝 허용 벡터."""
    n = len(labels)
    pre = [lab.partition("-")[0] for lab in labels]
    typ = [lab.partition("-")[2] for lab in labels]
    t = np.full((n, n), -np.inf, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if pre[i] in ("O", "E", "S"):
                if pre[j] == "O":
                    t[i, j] = b["transition_bias_background_stay"] if pre[i] == "O" else b["transition_bias_end_to_background"]
                elif pre[j] in ("B", "S"):
                    t[i, j] = b["transition_bias_background_to_start"] if pre[i] == "O" else b["transition_bias_end_to_start"]
            elif typ[i] == typ[j]:  # B, I
                if pre[j] == "I":
                    t[i, j] = b["transition_bias_inside_to_continue"]
                elif pre[j] == "E":
                    t[i, j] = b["transition_bias_inside_to_end"]
    start = np.array([0.0 if p in ("O", "B", "S") else -np.inf for p in pre], dtype=np.float32)
    end = np.array([0.0 if p in ("O", "E", "S") else -np.inf for p in pre], dtype=np.float32)
    return t, start, end


def logits(ids):
    """토큰 수만큼의 logits. 1024토큰 창을 겹쳐 돌리고 겹친 구간은 가운데 쪽 값을 씀."""
    out = None
    starts = list(range(0, max(len(ids) - STRIDE, 1), MAX_TOKENS - STRIDE))
    names = {i.name for i in _sess.get_inputs()}
    for k, s in enumerate(starts):
        e = min(s + MAX_TOKENS, len(ids))
        x = np.asarray([ids[s:e]], dtype=np.int64)
        feeds = {"input_ids": x}
        if "attention_mask" in names:
            feeds["attention_mask"] = np.ones_like(x)
        lg = _sess.run(["logits"], feeds)[0][0]
        if out is None:
            out = np.zeros((len(ids), lg.shape[-1]), dtype=np.float32)
        keep_from = s + (STRIDE // 2 if k > 0 else 0)
        keep_to = e - (STRIDE // 2 if k < len(starts) - 1 else 0)
        out[keep_from:keep_to] = lg[keep_from - s:keep_to - s]
    return out


def viterbi(lg):
    """제약 BIOES Viterbi: 로그 확률(log-softmax) 위에서 허용 전이만으로 최적 태그 열을 찾음."""
    lp = lg - lg.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    score = lp[0] + _start
    back = np.zeros(lp.shape, dtype=np.int32)
    for t in range(1, len(lp)):
        cand = score[:, None] + _trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    best = [int((score + _end).argmax())]
    for t in range(len(lp) - 1, 0, -1):
        best.append(int(back[t, best[-1]]))
    return best[::-1]


def detect(text):
    load()
    enc = _tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return []
    ents, cur = [], None
    for c, (s, e) in zip(viterbi(logits(ids)), offsets):
        prefix, _, typ = _labels[c].partition("-")
        if prefix in ("B", "S"):
            cur = {"type": typ, "start": s, "end": e}
            ents.append(cur)
        elif prefix in ("I", "E") and cur is not None:
            cur["end"] = max(cur["end"], e)
        if prefix in ("O", "E", "S"):
            cur = None
    return [{"label": LABEL_MAP[x["type"]], "start": x["start"], "end": x["end"]}
            for x in ents if x["type"] in LABEL_MAP]
