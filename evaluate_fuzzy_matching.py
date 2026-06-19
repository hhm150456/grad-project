"""
Evaluation script: Exact-match baseline vs. 3-rule fuzzy matcher
against a human-labeled skill-pair test set.

Usage:
    pip install scikit-learn --break-system-packages
    python evaluate_fuzzy_matching.py skill_pairs_test_set_100.csv

Outputs:
    - Console summary: accuracy, precision, recall, F1 for both methods
    - results_detailed.csv: per-pair predictions vs. gold labels
    - threshold_sweep.csv: F1 at thresholds 0.70-0.95 (for justifying default=0.80)
"""

import csv
import sys
from difflib import SequenceMatcher

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
except ImportError:
    sys.exit("Missing dependency. Run: pip install scikit-learn --break-system-packages")



def normalize(s: str) -> str:
    return s.strip().lower()


def exact_match(a: str, b: str) -> int:
    """Baseline: exact string match only."""
    return 1 if normalize(a) == normalize(b) else 0


def fuzzy_match(a: str, b: str, threshold: float = 0.80) -> int:
    """
    3-rule cascade matching the production _best_match_score() logic:
      1. Exact match            -> score 1.0
      2. Substring containment  -> score 0.9
      3. Sequence similarity    -> SequenceMatcher ratio
    Returns 1 if the resulting score >= threshold, else 0.
    """
    score = fuzzy_score(a, b)
    return 1 if score >= threshold else 0


def fuzzy_score(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()



def load_pairs(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "skill_a": row["skill_a"],
                "skill_b": row["skill_b"],
                "gold": int(row["is_match"]),
                "category": row.get("category", ""),
            })
    return rows



def evaluate(rows, default_threshold: float = 0.80):
    gold = [r["gold"] for r in rows]
    exact_preds = [exact_match(r["skill_a"], r["skill_b"]) for r in rows]
    fuzzy_preds = [fuzzy_match(r["skill_a"], r["skill_b"], default_threshold) for r in rows]

    results = {}
    for name, preds in [("Exact match (baseline)", exact_preds),
                         ("Fuzzy match (3-rule)", fuzzy_preds)]:
        results[name] = {
            "accuracy": accuracy_score(gold, preds),
            "precision": precision_score(gold, preds, zero_division=0),
            "recall": recall_score(gold, preds, zero_division=0),
            "f1": f1_score(gold, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(gold, preds).tolist(),
        }

    return results, exact_preds, fuzzy_preds


def threshold_sweep(rows, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)):
    gold = [r["gold"] for r in rows]
    sweep = []
    for t in thresholds:
        preds = [fuzzy_match(r["skill_a"], r["skill_b"], t) for r in rows]
        sweep.append({
            "threshold": t,
            "accuracy": accuracy_score(gold, preds),
            "precision": precision_score(gold, preds, zero_division=0),
            "recall": recall_score(gold, preds, zero_division=0),
            "f1": f1_score(gold, preds, zero_division=0),
        })
    return sweep


def print_summary(results):
    print("\n" + "=" * 60)
    print("EXACT MATCH vs FUZZY MATCH — EVALUATION SUMMARY")
    print("=" * 60)
    header = f"{'Method':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<26}{m['accuracy']:>10.3f}{m['precision']:>11.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}")
    print()
    for name, m in results.items():
        tn, fp, fn, tp = m["confusion_matrix"][0][0], m["confusion_matrix"][0][1], \
                          m["confusion_matrix"][1][0], m["confusion_matrix"][1][1]
        print(f"{name} confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print()


def print_threshold_sweep(sweep):
    print("=" * 60)
    print("THRESHOLD SENSITIVITY (fuzzy matcher)")
    print("=" * 60)
    header = f"{'Threshold':>10}{'Accuracy':>11}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print(header)
    print("-" * len(header))
    best = max(sweep, key=lambda r: r["f1"])
    for r in sweep:
        marker = "  <-- best F1" if r is best else ""
        print(f"{r['threshold']:>10.2f}{r['accuracy']:>11.3f}{r['precision']:>11.3f}"
              f"{r['recall']:>9.3f}{r['f1']:>8.3f}{marker}")
    print()


def write_detailed_csv(rows, exact_preds, fuzzy_preds, out_path="results_detailed.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["skill_a", "skill_b", "category", "gold",
                          "exact_pred", "fuzzy_pred", "fuzzy_score",
                          "exact_correct", "fuzzy_correct"])
        for r, ep, fp in zip(rows, exact_preds, fuzzy_preds):
            score = round(fuzzy_score(r["skill_a"], r["skill_b"]), 3)
            writer.writerow([
                r["skill_a"], r["skill_b"], r["category"], r["gold"],
                ep, fp, score,
                int(ep == r["gold"]), int(fp == r["gold"]),
            ])
    print(f"Detailed per-pair results written to: {out_path}")


def write_sweep_csv(sweep, out_path="threshold_sweep.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        for row in sweep:
            writer.writerow(row)
    print(f"Threshold sweep written to: {out_path}")



def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python evaluate_fuzzy_matching.py <skill_pairs.csv>")

    path = sys.argv[1]
    rows = load_pairs(path)
    print(f"Loaded {len(rows)} labeled skill pairs from {path}")

    results, exact_preds, fuzzy_preds = evaluate(rows, default_threshold=0.80)
    print_summary(results)

    sweep = threshold_sweep(rows)
    print_threshold_sweep(sweep)

    write_detailed_csv(rows, exact_preds, fuzzy_preds)
    write_sweep_csv(sweep)


if __name__ == "__main__":
    main()
