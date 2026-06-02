import argparse
import csv
import json
import math
import random
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

TYPE_ORDER = ["speech", "sound", "singing", "music"]
TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]


def read_csv(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["name"].strip()] = r
    return rows


def load_labels(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["name"].strip()
            label = r["label"].strip().lower()
            typ = r["type"].strip().lower()
            rows.append((name, 0 if label == "real" else 1, typ))
    return rows


def get_score(row):
    if "score" in row:
        return float(row["score"])
    if "logit_real" in row and "logit_fake" in row:
        lr = float(row["logit_real"])
        lf = float(row["logit_fake"])
        m = lr - lf
        return 1.0 / (1.0 + math.exp(-m))
    raise KeyError("score or logit columns not found")


def get_type_prob(row):
    if all(c in row for c in TYPE_COLS):
        q = np.asarray([float(row[c]) for c in TYPE_COLS], dtype=np.float64)
        s = q.sum()
        if s <= 1e-12:
            return np.ones(4, dtype=np.float64) / 4.0
        return q / s
    return np.ones(4, dtype=np.float64) / 4.0


def macro_by_type(y_true, y_pred, types):
    vals = []
    for t in TYPE_ORDER:
        idx = [i for i, x in enumerate(types) if x == t]
        if not idx:
            vals.append(0.0)
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        vals.append(f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0))
    return float(np.mean(vals)), vals


def all95_score(avg, per_type, floor=0.95, penalty=2.0):
    deficits = [max(0.0, floor - float(x)) for x in per_type]
    return float(avg) - penalty * float(np.mean(deficits))


def eval_params(s_base, s_ufm, q, y, types, alpha, bias, thresh, floor, penalty):
    # type-specific mixture first, then soft type posterior marginalization.
    # score = P(real). predict real if score >= threshold.
    alpha = np.asarray(alpha, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    per_type_scores = alpha[None, :] * s_base[:, None] + (1.0 - alpha[None, :]) * s_ufm[:, None]
    per_type_scores = np.clip(per_type_scores + bias[None, :], 1e-6, 1.0 - 1e-6)
    score = (q * per_type_scores).sum(axis=1)
    pred = np.where(score >= thresh, 0, 1)
    avg, per = macro_by_type(y.tolist(), pred.tolist(), types)
    obj = all95_score(avg, per, floor=floor, penalty=penalty)
    return obj, avg, per, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufm_csv", required=True)
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--random_trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ufm = read_csv(args.ufm_csv)
    base = read_csv(args.baseline_csv)
    labels = load_labels(args.label_csv)

    s_base, s_ufm, q_list, y, types, names = [], [], [], [], [], []
    missing = 0
    for name, lab, typ in labels:
        if name not in ufm or name not in base:
            missing += 1
            continue
        names.append(name)
        y.append(lab)
        types.append(typ)
        s_ufm.append(get_score(ufm[name]))
        s_base.append(get_score(base[name]))
        q_list.append(get_type_prob(ufm[name]))

    if not names:
        raise RuntimeError("No matched rows among ufm_csv, baseline_csv, and label_csv")

    s_base = np.asarray(s_base, dtype=np.float64)
    s_ufm = np.asarray(s_ufm, dtype=np.float64)
    q = np.stack(q_list, axis=0).astype(np.float64)
    y = np.asarray(y, dtype=np.int64)

    print("Matched:", len(names), "Missing:", missing)

    # Seed with sensible defaults: singing mostly baseline; weak3 mostly UFM.
    candidates = [
        ([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ([0.1, 0.1, 1.0, 0.1], [0.0, 0.0, 0.0, 0.0]),
        ([0.2, 0.2, 1.0, 0.2], [0.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0]),
    ]

    best = {"obj": -999, "avg": -1, "per_type": None, "alpha": None, "bias": None, "threshold": None}

    def try_one(alpha, bias):
        nonlocal best
        for th in np.linspace(0.05, 0.95, 181):
            obj, avg, per, _ = eval_params(s_base, s_ufm, q, y, types, alpha, bias, th, args.floor, args.penalty)
            if obj > best["obj"]:
                best.update({
                    "obj": float(obj),
                    "avg": float(avg),
                    "per_type": [float(x) for x in per],
                    "alpha": [float(x) for x in alpha],
                    "bias": [float(x) for x in bias],
                    "threshold": float(th),
                })

    for alpha, bias in candidates:
        try_one(alpha, bias)

    # Random search around plausible region.
    # alpha_t: 1 means baseline, 0 means UFM.
    for _ in range(args.random_trials):
        alpha = np.array([
            np.random.beta(1.5, 4.0),  # speech: prefer UFM but allow baseline
            np.random.beta(1.5, 4.0),  # sound
            np.random.beta(6.0, 1.5),  # singing: prefer baseline
            np.random.beta(1.5, 4.0),  # music
        ], dtype=np.float64)
        bias = np.random.uniform(-0.08, 0.08, size=4)
        try_one(alpha, bias)

    out = {
        "type_order": TYPE_ORDER,
        "alpha": best["alpha"],
        "bias": best["bias"],
        "threshold": best["threshold"],
        "objective_all95": best["obj"],
        "dev_macro_f1": best["avg"],
        "dev_per_type_f1": best["per_type"],
        "formula": "score=sum_t q_t * clip(alpha_t*score_baseline + (1-alpha_t)*score_ufm + bias_t); real if score>=threshold",
        "floor": args.floor,
        "penalty": args.penalty,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved:", args.out_json)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
