"""탐지 모델(atonlee KoELECTRA-small) 추가 학습 (worklog 31장). GPU 가 있으면 GPU, 없으면 CPU.

    python scripts/ft_train.py --out outputs/models/atonlee-ft-v1            # 기본: 3에폭, lr 3e-5, 길이 160
    PII_MODEL_ATONLEE=outputs/models/atonlee-ft-v1 python ...                 # 학습한 모델로 탐지 (hf_ner.py)

데이터: outputs/ft/train.jsonl, valid.jsonl ({text, entities:[{label,start,end}]}, scripts/ft_make_data.py)
라벨: 우리 7종 → 모델 태그 (NAME, PHONE, …) BIO. 모델의 다른 태그(DATE, ORG …)는 우리가 안 쓰므로 O 로 학습
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
BASE = "atonlee/koelectra-ko-pii-ner"
TYPE = {"person": "NAME", "phone_number": "PHONE", "email": "EMAIL", "address": "ADDRESS",
        "resident_registration_number": "RRN", "account_number": "ACCOUNT_NUMBER", "card_number": "CARD_NUMBER"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(tok, label2id, d, max_len):
    """글자 구간 → 토큰 BIO. 탐지 때처럼 add_special_tokens=False 로 자르고 CLS/SEP 를 직접 붙임 (hf_ner.tags 와 같게)"""
    enc = tok(d["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"][: max_len - 2], enc["offset_mapping"][: max_len - 2]
    labels = []
    for s, e in offs:
        lab = "O"
        for ent in d["entities"]:
            if s < ent["end"] and ent["start"] < e:
                first = s <= ent["start"] or not any(ps < ent["end"] and ent["start"] < pe for ps, pe in offs[:len(labels)])
                lab = ("B-" if first else "I-") + TYPE[ent["label"]]
                break
        labels.append(label2id[lab])
    return [tok.cls_token_id] + ids + [tok.sep_token_id], [-100] + labels + [-100]


def batches(rows, bs, shuffle, pad_id):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        L = max(len(x) for x, _ in chunk)
        x = torch.full((len(chunk), L), pad_id)
        y = torch.full((len(chunk), L), -100)
        m = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):
            x[k, :len(a)], y[k, :len(b)], m[k, :len(a)] = torch.tensor(a), torch.tensor(b), 1
        yield x, y, m


def evaluate(model, rows, dev, pad_id, o_id):
    """토큰 단위: 개인정보 토큰(O 아님) 정밀도·재현율"""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y, m in batches(rows, 64, False, pad_id):
            p = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.argmax(-1).cpu()
            keep = y != -100
            gp, pp = (y != o_id) & keep, (p != o_id) & keep
            tp += (gp & pp & (p == y)).sum().item()
            fp += (pp & ~(gp & (p == y))).sum().item()
            fn += (gp & ~(pp & (p == y))).sum().item()
    model.train()
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / max(P + R, 1e-9), 4)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "outputs" / "ft"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=160)
    a = ap.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(BASE, dtype=torch.float32).to(dev)
    label2id = model.config.label2id
    train = [encode(tok, label2id, d, a.max_len) for d in read(Path(a.data) / "train.jsonl")]
    valid = [encode(tok, label2id, d, a.max_len) for d in read(Path(a.data) / "valid.jsonl")]
    o_id = label2id["O"]
    print(f"장치 {dev} · 학습 {len(train)} · 검증 {len(valid)} · 시작 전 검증 {evaluate(model, valid, dev, tok.pad_token_id, o_id)}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    model.train()
    t0, step, log = time.time(), 0, []
    for ep in range(1, a.epochs + 1):
        total = 0.0
        for x, y, m in batches(train, a.bs, True, tok.pad_token_id):
            loss = model(input_ids=x.to(dev), attention_mask=m.to(dev), labels=y.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            total += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"  스텝 {step}/{steps} · 손실 {total / (step - (ep - 1) * math.ceil(len(train) / a.bs)):.4f} · "
                      f"{time.time() - t0:.0f}s", flush=True)
        v = evaluate(model, valid, dev, tok.pad_token_id, o_id)
        log.append({"epoch": ep, "valid": v, "sec": round(time.time() - t0)})
        print(f"에폭 {ep}: 검증 {v} · {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "device": dev, "log": log}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
