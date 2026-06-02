import argparse
import csv
import json
from pathlib import Path
import numpy as np

TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]


def read_csv(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["name"].strip()] = r
    return rows


def get_score(row):
    if "score" in row:
        return float(row["score"])
    if "logit_real" in row and "logit_fake" in row:
        m = float(row["logit_real"]) - float(row["logit_fake"])
        return 1.0 / (1.0 + np.exp(-m))
    raise KeyError("score or logit columns not found")


def get_type_prob(row):
    if all(c in row for c in TYPE_COLS):
        q = np.asarray([float(row[c]) for c in TYPE_COLS], dtype=np.float64)
        s = q.sum()
        if s <= 1e-12:
            return np.ones(4, dtype=np.float64) / 4.0
        return q / s
    return np.ones(4, dtype=np.float64) / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufm_csv", required=True)
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--calib_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    ufm = read_csv(args.ufm_csv)
    base = read_csv(args.baseline_csv)
    with open(args.calib_json, "r", encoding="utf-8") as f:
        calib = json.load(f)

    alpha = np.asarray(calib["alpha"], dtype=np.float64)
    bias = np.asarray(calib["bias"], dtype=np.float64)
    threshold = float(calib["threshold"])

    names = sorted(ufm.keys())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    missing = 0

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "predict"])

        for name in names:
            if name not in base:
                missing += 1
                continue
            su = get_score(ufm[name])
            sb = get_score(base[name])
            q = get_type_prob(ufm[name])
            per_type_scores = alpha * sb + (1.0 - alpha) * su + bias
            per_type_scores = np.clip(per_type_scores, 1e-6, 1.0 - 1e-6)
            score = float((q * per_type_scores).sum())
            pred = "real" if score >= threshold else "fake"
            writer.writerow([name, pred])

    print("Saved:", args.out_csv)
    print("Missing baseline rows:", missing)
    print("threshold:", threshold)


if __name__ == "__main__":
    main()
