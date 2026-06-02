import argparse
import csv
import numpy as np
from sklearn.metrics import f1_score

TYPE_ORDER = ["speech", "sound", "singing", "music"]

def read_score(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["name"].strip()] = r
    return rows

def read_label(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "name": r["name"].strip(),
                "label": 0 if r["label"].strip().lower() == "real" else 1,
                "type": r["type"].strip().lower()
            })
    return rows

def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    args = ap.parse_args()

    score_rows = read_score(args.score_csv)
    label_rows = read_label(args.label_csv)

    y, score, types = [], [], []

    for r in label_rows:
        name = r["name"]
        if name not in score_rows:
            continue
        y.append(r["label"])
        score.append(float(score_rows[name]["score"]))
        types.append(r["type"])

    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)
    types = np.asarray(types)

    print("Matched:", len(y))

    print("\n[Global threshold]")
    best = (-1, None)
    for th in np.linspace(0.01, 0.99, 197):
        pred = np.where(score >= th, 0, 1)
        f = macro_f1(y, pred)
        if f > best[0]:
            best = (f, th)
    print("best_global_f1:", best[0], "threshold:", best[1])

    print("\n[Per-type threshold]")
    per = []
    for t in TYPE_ORDER:
        idx = (types == t)
        yt = y[idx]
        st = score[idx]

        best_t = (-1, None)
        for th in np.linspace(0.01, 0.99, 197):
            pred = np.where(st >= th, 0, 1)
            f = macro_f1(yt, pred)
            if f > best_t[0]:
                best_t = (f, th)

        per.append(best_t[0])
        print(f"{t:8s} best_f1={best_t[0]:.6f} threshold={best_t[1]:.4f} n={idx.sum()}")

    print("\nper_type_mean:", float(np.mean(per)))
    print("min_type:", float(np.min(per)))

if __name__ == "__main__":
    main()
