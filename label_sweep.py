"""Evaluate label leakage from one and three FedMRN client uploads."""

import argparse
import json
from pathlib import Path

import torch

from audit import HERE, infer_label, load_sample, make_observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.count < 1 or args.rounds < 2:
        parser.error("count must be >=1 and rounds >=2")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for index in range(args.count):
        image, label = load_sample(index)
        records = make_observations(image, label, args.rounds, 0.1,
                                    args.seed + index * 100, device)
        raw = infer_label(records[0]["raw_gradients"]["classifier.weight"])
        single = infer_label(records[0]["wire_estimate"]["classifier.weight"])
        combined = sum((r["wire_estimate"]["classifier.weight"] for r in records))
        multi = infer_label(combined)
        rows.append({"index": index, "true_label": label,
                     "raw_gradient_prediction": raw, "single_wire_prediction": single,
                     "multi_wire_prediction": multi})
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{args.count}", flush=True)
    summary = {"sample_count": args.count, "rounds": args.rounds, "seed": args.seed,
               "device": device, "rows": rows,
               "correct_raw": sum(r["raw_gradient_prediction"] == r["true_label"] for r in rows),
               "correct_single_wire": sum(r["single_wire_prediction"] == r["true_label"] for r in rows),
               "correct_multi_wire": sum(r["multi_wire_prediction"] == r["true_label"] for r in rows)}
    out = HERE / "results" / "label_sweep.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
