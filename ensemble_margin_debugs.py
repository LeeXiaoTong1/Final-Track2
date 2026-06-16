#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
import numpy as np


def read_debug(path):
    data = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["name"].strip()
            score = float(r["score"])
            th = float(r["threshold"])
            data[name] = score - th
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="append", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--trim", action="store_true")
    args = ap.parse_args()

    arrs = [read_debug(p) for p in args.debug]
    names = sorted(set.intersection(*[set(x.keys()) for x in arrs]))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "predict"])

        for name in names:
            m = np.array([d[name] for d in arrs], dtype=np.float64)

            if args.trim and len(m) >= 5:
                m = np.sort(m)[1:-1]

            margin = float(m.mean())
            pred = "real" if margin >= 0 else "fake"
            writer.writerow([name, pred])

    print("Saved:", args.out_csv)
    print("Rows:", len(names))


if __name__ == "__main__":
    main()
