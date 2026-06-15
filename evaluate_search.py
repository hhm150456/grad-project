"""
evaluate_search.py
Evaluate retrieval quality and latency for the job-search API's ranking methods:

  - bm25            -> POST /bm25-search
  - hybrid          -> POST /hybrid-search  (BM25 + kNN merged via RRF)
  - similarity      -> POST /search-query (embed) + POST /similarity (kNN)
  - hybrid_rerank   -> POST /hybrid-search (candidate pool) + POST /rerank (cross-encoder)

For each method and each labelled query, computes:
  - NDCG@10
  - Recall@10
  - Precision@10
  - MRR (Mean Reciprocal Rank, considered within the top-10 only)

And measures end-to-end request latency (all HTTP calls for that method/query),
reporting the average and P95 across queries x repetitions.

USAGE
-----
    python evaluate_search.py --eval-file eval_dataset.json

    python evaluate_search.py \\
        --base-url http://localhost:8000 \\
        --eval-file eval_dataset.json \\
        --k 10 \\
        --rerank-pool 30 \\
        --reps 3 \\
        --output results.csv

EVAL DATASET FORMAT (eval_dataset.json)
----------------------------------------
A JSON list of query objects. `relevant` maps job_id -> relevance grade
(integer >= 1). Grades are used for NDCG; for Precision/Recall/MRR any
job_id present in `relevant` with grade >= 1 is treated as relevant.

[
  {
    "query": "python data engineer remote",
    "relevant": {
      "123": 3,
      "456": 2,
      "789": 1
    }
  },
  {
    "query": "entry level ai engineer with python",
    "relevant": {
      "321": 3,
      "654": 1
    }
  }
]

A small example file is written automatically if --eval-file does not exist
and --write-example is passed.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    gains = [relevant.get(jid, 0) for jid in ranked_ids[:k]]
    dcg = dcg_at_k(gains, k)

    ideal_gains = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for jid in ranked_ids[:k] if relevant.get(jid, 0) > 0)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    total_relevant = sum(1 for v in relevant.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for jid in ranked_ids[:k] if relevant.get(jid, 0) > 0)
    return hits / total_relevant


def mrr_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    for i, jid in enumerate(ranked_ids[:k]):
        if relevant.get(jid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Method runners — each returns (ranked_job_ids, latency_seconds)
# ─────────────────────────────────────────────────────────────────────────────

class SearchClient:
    def __init__(self, base_url: str, k: int, rerank_pool: int):
        self.base_url = base_url.rstrip("/")
        self.k = k
        self.rerank_pool = rerank_pool
        self.session = requests.Session()

    def _post(self, path: str, payload: dict) -> dict:
        resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def run_bm25(self, query: str) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        data = self._post("/bm25-search", {"query": query, "k": self.k})
        latency = time.perf_counter() - t0
        ranked_ids = [r["job_id"] for r in data["results"]]
        return ranked_ids, latency

    def run_hybrid(self, query: str) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        data = self._post("/hybrid-search", {"query": query, "k": self.k})
        latency = time.perf_counter() - t0
        ranked_ids = [r["job_id"] for r in data["results"]]
        return ranked_ids, latency

    def run_similarity(self, query: str) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        emb_data = self._post("/search-query", {"query": query})
        embedding = emb_data["embedding"]
        sim_data = self._post(
            "/similarity",
            {
                "query_embedding": embedding,
                "k": self.k,
                "num_candidates": max(100, self.k * 10),
            },
        )
        latency = time.perf_counter() - t0
        ranked_ids = [r["job_id"] for r in sim_data["results"]]
        return ranked_ids, latency

    def run_hybrid_rerank(self, query: str) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        hybrid_data = self._post("/hybrid-search", {"query": query, "k": self.rerank_pool})

        jobs_payload = [
            {
                "job_id": r["job_id"],
                "title": r["job_title"],
                "description": r["job_description"],
            }
            for r in hybrid_data["results"]
        ]

        if not jobs_payload:
            latency = time.perf_counter() - t0
            return [], latency

        rerank_data = self._post("/rerank", {"query": query, "jobs": jobs_payload})
        latency = time.perf_counter() - t0

        ranked_ids = [r["job_id"] for r in rerank_data["results"]][: self.k]
        return ranked_ids, latency


METHODS = {
    "bm25": SearchClient.run_bm25,
    "hybrid": SearchClient.run_hybrid,
    "similarity": SearchClient.run_similarity,
    "hybrid_rerank": SearchClient.run_hybrid_rerank,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "query" not in item or "relevant" not in item:
            raise ValueError(
                f"Each eval item needs 'query' and 'relevant' keys, got: {item!r}"
            )
        # Normalize relevant keys to strings (job_ids are strings in the API)
        item["relevant"] = {str(k): int(v) for k, v in item["relevant"].items()}

    return data


def write_example_dataset(path: Path) -> None:
    example = [
        {
            "query": "python data engineer remote",
            "relevant": {"123": 3, "456": 2, "789": 1},
        },
        {
            "query": "entry level ai engineer with python",
            "relevant": {"321": 3, "654": 1},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"Wrote example eval dataset to {path}. "
          f"Replace job_ids/relevance grades with real judgments from your index.")


def evaluate(client: SearchClient, dataset: list[dict], k: int, reps: int) -> dict:
    """
    Runs every method against every query `reps` times.

    Ranking-quality metrics are computed from the first repetition only
    (results should be deterministic); latency is pooled across all reps.
    """
    summary = {}

    for method_name, runner in METHODS.items():
        ndcgs, precisions, recalls, mrrs = [], [], [], []
        latencies = []
        errors = 0

        for item in dataset:
            query = item["query"]
            relevant = item["relevant"]

            for rep in range(reps):
                try:
                    ranked_ids, latency = runner(client, query)
                except Exception as exc:
                    errors += 1
                    print(f"  [{method_name}] ERROR on query={query!r} (rep {rep}): {exc}",
                          file=sys.stderr)
                    continue

                latencies.append(latency)

                if rep == 0:
                    ndcgs.append(ndcg_at_k(ranked_ids, relevant, k))
                    precisions.append(precision_at_k(ranked_ids, relevant, k))
                    recalls.append(recall_at_k(ranked_ids, relevant, k))
                    mrrs.append(mrr_at_k(ranked_ids, relevant, k))

        summary[method_name] = {
            "ndcg@10": statistics.mean(ndcgs) if ndcgs else 0.0,
            "recall@10": statistics.mean(recalls) if recalls else 0.0,
            "precision@10": statistics.mean(precisions) if precisions else 0.0,
            "mrr": statistics.mean(mrrs) if mrrs else 0.0,
            "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
            "p95_latency_ms": percentile(latencies, 95) * 1000 if latencies else 0.0,
            "queries_evaluated": len(ndcgs),
            "errors": errors,
        }

    return summary


def print_summary(summary: dict) -> None:
    headers = ["method", "NDCG@10", "Recall@10", "Precision@10", "MRR",
               "AvgLatency(ms)", "P95Latency(ms)", "queries", "errors"]
    rows = []
    for method, m in summary.items():
        rows.append([
            method,
            f"{m['ndcg@10']:.4f}",
            f"{m['recall@10']:.4f}",
            f"{m['precision@10']:.4f}",
            f"{m['mrr']:.4f}",
            f"{m['avg_latency_ms']:.1f}",
            f"{m['p95_latency_ms']:.1f}",
            str(m["queries_evaluated"]),
            str(m["errors"]),
        ])

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def write_csv(summary: dict, path: Path) -> None:
    fieldnames = ["method", "ndcg@10", "recall@10", "precision@10", "mrr",
                  "avg_latency_ms", "p95_latency_ms", "queries_evaluated", "errors"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, m in summary.items():
            row = {"method": method, **m}
            writer.writerow(row)
    print(f"\nWrote results to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate job search ranking methods.")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the running FastAPI app (default: http://localhost:8000)")
    parser.add_argument("--eval-file", default="eval_dataset.json",
                        help="Path to the labelled eval dataset JSON (default: eval_dataset.json)")
    parser.add_argument("--k", type=int, default=10,
                        help="Cutoff for NDCG/Recall/Precision/MRR and number of results returned (default: 10)")
    parser.add_argument("--rerank-pool", type=int, default=30,
                        help="Candidate pool size fetched from /hybrid-search before reranking (default: 30)")
    parser.add_argument("--reps", type=int, default=3,
                        help="Number of repetitions per query for latency measurement (default: 3)")
    parser.add_argument("--output", default=None,
                        help="Optional path to write a CSV summary")
    parser.add_argument("--write-example", action="store_true",
                        help="If --eval-file doesn't exist, write an example dataset there and exit")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)

    if not eval_path.exists():
        if args.write_example:
            write_example_dataset(eval_path)
            return
        print(f"Eval file not found: {eval_path}\n"
              f"Run again with --write-example to generate a template.", file=sys.stderr)
        sys.exit(1)

    dataset = load_eval_dataset(eval_path)
    print(f"Loaded {len(dataset)} labelled queries from {eval_path}")

    client = SearchClient(base_url=args.base_url, k=args.k, rerank_pool=args.rerank_pool)

    print(f"Evaluating against {args.base_url} (k={args.k}, reps={args.reps}, "
          f"rerank_pool={args.rerank_pool})...\n")

    summary = evaluate(client, dataset, k=args.k, reps=args.reps)

    print()
    print_summary(summary)

    if args.output:
        write_csv(summary, Path(args.output))


if __name__ == "__main__":
    main()
