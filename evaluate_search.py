"""
evaluate_search_native.py
Native (in-process) evaluation of retrieval quality and latency for the
job-search ranking methods. This script does NOT make HTTP calls — it
imports and calls the underlying Python functions directly
(elastic_service, embedding_service, reranker_service), the same way the
FastAPI endpoints do internally.

Follows the "after qrels" workflow:

  Step 2: Run each system on ALL queries
            -> --run, one call per system (bm25 / dense / hybrid / hybrid_rerank)
  Step 3: Save ranked outputs to CSV: query_id, job_id, rank, score
            -> written to --rankings-dir as <system>_rankings.csv
  Step 4: Evaluate using qrels (compare saved rankings vs ground truth)
            -> --evaluate, reads --rankings-dir + --eval-file
  Step 5: Per-system metrics table
            -> NDCG@10, Recall@10, Precision@10, MRR, MAP
  Step 6: Latency evaluation
            -> avg / P95 / P99 latency, measured during --run (Step 2)

Systems evaluated:
  - bm25            -> elastic_service.bm25_search_jobs(query, k)
  - dense            -> embedding_service.generate_query_embedding(query)
                       + elastic_service.knn_search_jobs(embedding, k, num_candidates)
  - hybrid          -> embedding_service.generate_query_embedding(query)
                       + elastic_service.hybrid_search_jobs(query, embedding, k)
  - hybrid_rerank   -> hybrid (candidate pool) + reranker_service.rerank_jobs(query, jobs)

("similarity" from earlier versions of this script has been renamed "dense"
to match the workflow's naming: BM25 / Dense retrieval (JobBERT) / Hybrid /
Hybrid + Reranker.)

For each system and each labelled query, computes:
  - NDCG@10
  - Recall@10
  - Precision@10
  - MRR  (Mean Reciprocal Rank, considered within the top-k only)
  - MAP  (Mean Average Precision, considered within the top-k only)

And measures wall-clock latency for the in-process call(s) made during
--run, reporting average, P95, and P99 across queries x repetitions.

USAGE
-----
    # Step 2 + 3: run all systems on all queries, save ranked outputs
    python evaluate_search_native.py --run \\
        --eval-file eval_dataset.json \\
        --rankings-dir rankings/ \\
        --k 10 --rerank-pool 30 --reps 3

    # Step 4 + 5 + 6: evaluate saved rankings against qrels, report metrics + latency
    python evaluate_search_native.py --evaluate \\
        --eval-file eval_dataset.json \\
        --rankings-dir rankings/ \\
        --k 10 \\
        --output results.csv

    # Or do both in one go (run, then evaluate immediately)
    python evaluate_search_native.py --run --evaluate \\
        --eval-file eval_dataset.json \\
        --rankings-dir rankings/ \\
        --output results.csv

This script must be run from (or have on sys.path) the same directory as
main.py / elastic_service.py / embedding_service.py / reranker_service.py,
since it imports those modules directly. --run will load TechWolf/JobBERT-v2
and the cross-encoder reranker model into memory — the same models used by
the running app — so make sure HF_TOKEN / pagefile / memory considerations
from the app also apply here. --evaluate alone does not need those models,
since it only reads previously saved ranking CSVs.

RANKED OUTPUT FORMAT (rankings/<system>_rankings.csv) — Step 3
----------------------------------------------------------------
    query_id,job_id,rank,score
    Q1,J10,1,12.4
    Q1,J55,2,11.8
    Q1,J22,3,11.1

`query_id` here is the 0-based index of the query within --eval-file (since
eval_dataset.json stores query text, not an explicit id — pass a qrels-style
file with ids via --queries-file if you want your own Q1/Q2/... ids
reflected in this CSV; see --queries-file below).

EVAL DATASET FORMAT — option A: judged-pool CSV (--eval-file *.csv)
-----------------------------------------------------------------------
Columns (header row required, extra columns are ignored):
    query_id,query,job_id,normalized_job_title,job_description,sources,relevance,job_title

Relevance scale (graded; used directly for NDCG/MAP, and as >=1 for the
binary "is this relevant" check in Precision/Recall/MRR):
    2 = Highly Relevant   — strongly satisfies search intent
    1 = Partially Relevant — some overlap, not an ideal match
    0 = Not Relevant      — does not satisfy the search intent

Example:
    query_id,query,job_id,normalized_job_title,job_description,sources,relevance,job_title
    Q1,python data engineer remote,123,Data Engineer,Build ETL...,bm25;dense,2,Python Data Engineer
    Q1,python data engineer remote,456,Data Analyst,SQL reporting...,bm25,1,Data Analyst
    Q1,python data engineer remote,789,Frontend Engineer,React work...,hybrid,0,Frontend Engineer

Rows are grouped by query_id; rows with a blank "relevance" (not yet
judged) are skipped with a warning rather than treated as 0.

This is exactly the format produced by build_qrels_pool.py's
pool_to_judge.csv once you've filled in the "relevance" column by hand —
no separate --convert step is needed, just pass that judged CSV straight
to --eval-file here.

EVAL DATASET FORMAT — option B: eval_dataset.json (--eval-file *.json)
-----------------------------------------------------------------------
A JSON list of query objects. `relevant` maps job_id -> relevance grade
(integer, same 0/1/2 scale as above). Grades are used for NDCG/MAP; for
Precision/Recall/MRR any job_id present in `relevant` with grade >= 1 is
treated as relevant.

[
  {
    "query": "python data engineer remote",
    "relevant": {
      "123": 2,
      "456": 1,
      "789": 0
    }
  },
  {
    "query": "entry level ai engineer with python",
    "relevant": {
      "321": 2,
      "654": 1
    }
  }
]

Optionally include a "query_id" field per item (e.g. "Q1") — if present,
it's used as the query_id column in the saved ranking CSVs instead of a
0-based index. build_qrels_pool.py's --convert step already writes
eval_dataset.json without query_id; add it manually if you want your
original Q1/Q2/.../Q100 ids preserved end-to-end.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

# Make sure the project root (where main.py etc. live) is importable, even if
# this script is run from a different working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def map_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """Mean Average Precision @ k (averaged later across queries by the caller)."""
    total_relevant = sum(1 for v in relevant.values() if v > 0)
    if total_relevant == 0:
        return 0.0

    hits = 0
    precisions_at_hits = []
    for i, jid in enumerate(ranked_ids[:k]):
        if relevant.get(jid, 0) > 0:
            hits += 1
            precisions_at_hits.append(hits / (i + 1))

    if not precisions_at_hits:
        return 0.0

    return sum(precisions_at_hits) / total_relevant


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[idx]


# ─────────────────────────────────────────────────────────────────────────────
# System runners — each returns (ranked_results, latency_seconds), where
# ranked_results is a list of {"job_id": ..., "score": ...} in rank order.
# Heavy imports (elastic_service / embedding_service / reranker_service) are
# done lazily inside each function so --evaluate-only runs don't need ES or
# the ML models installed.
# ─────────────────────────────────────────────────────────────────────────────

def run_bm25(query: str, k: int, rerank_pool: int) -> tuple[list[dict], float]:
    from elastic_service import bm25_search_jobs

    t0 = time.perf_counter()
    results = bm25_search_jobs(query=query, k=k)
    latency = time.perf_counter() - t0
    return [{"job_id": r["job_id"], "score": r["score"]} for r in results], latency


def run_dense(query: str, k: int, rerank_pool: int) -> tuple[list[dict], float]:
    from elastic_service import knn_search_jobs
    from embedding_service import generate_query_embedding

    t0 = time.perf_counter()
    embedding = generate_query_embedding(query, seeker_embedding=None)
    results = knn_search_jobs(query_embedding=embedding, k=k, num_candidates=max(100, k * 10))
    latency = time.perf_counter() - t0
    return [{"job_id": r["job_id"], "score": r["score"]} for r in results], latency


def run_hybrid(query: str, k: int, rerank_pool: int) -> tuple[list[dict], float]:
    from elastic_service import hybrid_search_jobs
    from embedding_service import generate_query_embedding

    t0 = time.perf_counter()
    embedding = generate_query_embedding(query, seeker_embedding=None)
    results = hybrid_search_jobs(query=query, query_embedding=embedding, k=k)
    latency = time.perf_counter() - t0
    return [{"job_id": r["job_id"], "score": r["hybrid_score"]} for r in results], latency


def run_hybrid_rerank(query: str, k: int, rerank_pool: int) -> tuple[list[dict], float]:
    from elastic_service import hybrid_search_jobs
    from embedding_service import generate_query_embedding
    from reranker_service import rerank_jobs

    t0 = time.perf_counter()
    embedding = generate_query_embedding(query, seeker_embedding=None)
    candidates = hybrid_search_jobs(query=query, query_embedding=embedding, k=rerank_pool)

    if not candidates:
        latency = time.perf_counter() - t0
        return [], latency

    jobs_payload = [
        {
            "job_id": c["job_id"],
            "title": c["job_title"],
            "description": c["job_description"],
        }
        for c in candidates
    ]

    reranked = rerank_jobs(query=query, jobs=jobs_payload)
    latency = time.perf_counter() - t0

    top = reranked[:k]
    return [{"job_id": r["job_id"], "score": r["rerank_score"]} for r in top], latency


SYSTEMS = {
    "bm25": run_bm25,
    "dense": run_dense,
    "hybrid": run_hybrid,
    "hybrid_rerank": run_hybrid_rerank,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_dataset_json(path: Path) -> list[dict]:
    """
    Returns a list of {"query_id": ..., "query": ..., "relevant": {...}} dicts.
    If an item has no "query_id", a 0-based index ("0", "1", ...) is used.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for i, item in enumerate(data):
        if "query" not in item or "relevant" not in item:
            raise ValueError(
                f"Each eval item needs 'query' and 'relevant' keys, got: {item!r}"
            )
        items.append({
            "query_id": str(item.get("query_id", i)),
            "query": item["query"],
            "relevant": {str(k): int(v) for k, v in item["relevant"].items()},
        })

    return items


def load_eval_dataset_csv(path: Path) -> list[dict]:
    """
    Loads qrels from a judged-pool CSV with columns:
        query_id,query,job_id,normalized_job_title,job_description,sources,relevance,job_title

    Only query_id, query, job_id, and relevance are actually used here;
    normalized_job_title / job_description / sources / job_title are
    metadata from the pooling step and are ignored for evaluation purposes.

    Relevance scale (graded, used directly for NDCG/MAP and as >=1 for
    Precision/Recall/MRR "is this relevant" checks):
        2 = Highly Relevant
        1 = Partially Relevant
        0 = Not Relevant

    Rows with an empty/missing relevance value are skipped (treated as
    not yet judged) with a warning, rather than silently counted as 0.

    Returns a list of {"query_id": ..., "query": ..., "relevant": {...}}
    dicts, one per unique query_id, in first-seen order.
    """
    required = {"query_id", "query", "job_id", "relevance"}

    by_query_order: list[str] = []
    query_text_by_id: dict[str, str] = {}
    relevant_by_query: dict[str, dict[str, int]] = {}
    skipped = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            rel_str = (row.get("relevance") or "").strip()
            if rel_str == "":
                skipped += 1
                continue
            try:
                rel = int(rel_str)
            except ValueError:
                print(f"WARNING: non-integer relevance {rel_str!r} for "
                      f"query_id={row['query_id']} job_id={row['job_id']}, skipping row.",
                      file=sys.stderr)
                skipped += 1
                continue
            if rel not in (0, 1, 2):
                print(f"WARNING: relevance {rel} for query_id={row['query_id']} "
                      f"job_id={row['job_id']} is outside the expected 0-2 scale; "
                      f"using it as-is.", file=sys.stderr)

            qid = row["query_id"].strip()
            job_id = row["job_id"].strip()

            if qid not in relevant_by_query:
                relevant_by_query[qid] = {}
                query_text_by_id[qid] = row["query"]
                by_query_order.append(qid)

            relevant_by_query[qid][job_id] = rel

    if skipped:
        print(f"NOTE: skipped {skipped} unjudged/invalid row(s) in {path}.", file=sys.stderr)

    return [
        {"query_id": qid, "query": query_text_by_id[qid], "relevant": relevant_by_query[qid]}
        for qid in by_query_order
    ]


def load_eval_dataset(path: Path) -> list[dict]:
    """
    Dispatches to the CSV or JSON loader based on file extension.
    """
    if path.suffix.lower() == ".csv":
        return load_eval_dataset_csv(path)
    return load_eval_dataset_json(path)


def write_example_dataset(path: Path) -> None:
    if path.suffix.lower() == ".csv":
        fieldnames = ["query_id", "query", "job_id", "normalized_job_title",
                      "job_description", "sources", "relevance", "job_title"]
        rows = [
            {"query_id": "Q1", "query": "python data engineer remote",
             "job_id": "123", "normalized_job_title": "Data Engineer",
             "job_description": "Build ETL pipelines in Python...",
             "sources": "bm25;dense", "relevance": "2", "job_title": "Python Data Engineer"},
            {"query_id": "Q1", "query": "python data engineer remote",
             "job_id": "456", "normalized_job_title": "Data Analyst",
             "job_description": "SQL reporting and dashboards...",
             "sources": "bm25", "relevance": "1", "job_title": "Data Analyst"},
            {"query_id": "Q1", "query": "python data engineer remote",
             "job_id": "789", "normalized_job_title": "Frontend Engineer",
             "job_description": "React and CSS work...",
             "sources": "hybrid", "relevance": "0", "job_title": "Frontend Engineer"},
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote example eval dataset (CSV) to {path}. "
              f"Replace job_ids/relevance grades with real judgments from your index.")
        return

    example = [
        {
            "query": "python data engineer remote",
            "relevant": {"123": 2, "456": 1, "789": 0},
        },
        {
            "query": "entry level ai engineer with python",
            "relevant": {"321": 2, "654": 1},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"Wrote example eval dataset (JSON) to {path}. "
          f"Replace job_ids/relevance grades with real judgments from your index.")


def run_systems_and_save_rankings(
    dataset: list[dict], k: int, rerank_pool: int, reps: int, rankings_dir: Path,
) -> dict:
    """
    Step 2: run every system on every query (repeated `reps` times for
    stable latency numbers).
    Step 3: persist the rank-ordered output of each system to
    <rankings_dir>/<system>_rankings.csv as query_id,job_id,rank,score
    (only the first rep's ranking is saved — reps exist purely to get a
    latency distribution, results are expected to be deterministic).

    Returns a dict of system_name -> list of per-call latencies (seconds),
    collected across all reps, for the Step 6 latency report.
    """
    rankings_dir.mkdir(parents=True, exist_ok=True)
    latencies_by_system: dict[str, list[float]] = {name: [] for name in SYSTEMS}

    for system_name, runner in SYSTEMS.items():
        print(f"Running system: {system_name}")
        rows = []
        errors = 0

        for item in dataset:
            query_id = item["query_id"]
            query = item["query"]

            for rep in range(reps):
                try:
                    ranked_results, latency = runner(query, k, rerank_pool)
                except Exception as exc:
                    errors += 1
                    print(f"  [{system_name}] ERROR on query_id={query_id} "
                          f"query={query!r} (rep {rep}): {exc}", file=sys.stderr)
                    continue

                latencies_by_system[system_name].append(latency)

                if rep == 0:
                    for rank, r in enumerate(ranked_results, start=1):
                        rows.append({
                            "query_id": query_id,
                            "job_id": r["job_id"],
                            "rank": rank,
                            "score": r["score"],
                        })

        out_path = rankings_dir / f"{system_name}_rankings.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "job_id", "rank", "score"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> saved {len(rows)} ranked rows to {out_path} ({errors} error(s))\n")

    return latencies_by_system


def load_rankings(path: Path) -> dict[str, list[str]]:
    """
    Reads a query_id,job_id,rank,score CSV and returns
    query_id -> [job_id, job_id, ...] ordered by rank ascending.
    """
    by_query: dict[str, list[tuple[int, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            by_query.setdefault(qid, []).append((int(row["rank"]), row["job_id"]))

    return {
        qid: [jid for _, jid in sorted(pairs, key=lambda p: p[0])]
        for qid, pairs in by_query.items()
    }


def evaluate_from_rankings(
    dataset: list[dict], k: int, rankings_dir: Path,
    latencies_by_system: dict[str, list[float]] | None = None,
) -> dict:
    """
    Step 4/5: compare each system's saved ranking CSV against the qrels
    (dataset's `relevant` field) and compute NDCG@10, Recall@10,
    Precision@10, MRR, MAP.
    Step 6: attach avg / P95 / P99 latency, if latencies_by_system is given
    (i.e. when --run was executed in the same invocation). If rankings were
    loaded from a previous --run, latency is reported as 0 / not available.
    """
    relevant_by_query = {item["query_id"]: item["relevant"] for item in dataset}
    summary = {}

    for system_name in SYSTEMS:
        rankings_path = rankings_dir / f"{system_name}_rankings.csv"
        if not rankings_path.exists():
            print(f"WARNING: {rankings_path} not found, skipping {system_name}. "
                  f"Run with --run first.", file=sys.stderr)
            continue

        rankings = load_rankings(rankings_path)

        ndcgs, precisions, recalls, mrrs, maps = [], [], [], [], []
        for query_id, relevant in relevant_by_query.items():
            ranked_ids = rankings.get(query_id, [])
            ndcgs.append(ndcg_at_k(ranked_ids, relevant, k))
            precisions.append(precision_at_k(ranked_ids, relevant, k))
            recalls.append(recall_at_k(ranked_ids, relevant, k))
            mrrs.append(mrr_at_k(ranked_ids, relevant, k))
            maps.append(map_at_k(ranked_ids, relevant, k))

        latencies = (latencies_by_system or {}).get(system_name, [])

        summary[system_name] = {
            "ndcg@10": statistics.mean(ndcgs) if ndcgs else 0.0,
            "recall@10": statistics.mean(recalls) if recalls else 0.0,
            "precision@10": statistics.mean(precisions) if precisions else 0.0,
            "mrr": statistics.mean(mrrs) if mrrs else 0.0,
            "map": statistics.mean(maps) if maps else 0.0,
            "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
            "p95_latency_ms": percentile(latencies, 95) * 1000 if latencies else 0.0,
            "p99_latency_ms": percentile(latencies, 99) * 1000 if latencies else 0.0,
            "queries_evaluated": len(ndcgs),
        }

    return summary


def print_summary(summary: dict) -> None:
    headers = ["system", "NDCG@10", "Recall@10", "Precision@10", "MRR", "MAP",
               "AvgLatency(ms)", "P95Latency(ms)", "P99Latency(ms)", "queries"]
    rows = []
    for system, m in summary.items():
        rows.append([
            system,
            f"{m['ndcg@10']:.4f}",
            f"{m['recall@10']:.4f}",
            f"{m['precision@10']:.4f}",
            f"{m['mrr']:.4f}",
            f"{m['map']:.4f}",
            f"{m['avg_latency_ms']:.1f}",
            f"{m['p95_latency_ms']:.1f}",
            f"{m['p99_latency_ms']:.1f}",
            str(m["queries_evaluated"]),
        ])

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def write_csv(summary: dict, path: Path) -> None:
    fieldnames = ["system", "ndcg@10", "recall@10", "precision@10", "mrr", "map",
                  "avg_latency_ms", "p95_latency_ms", "p99_latency_ms", "queries_evaluated"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for system, m in summary.items():
            row = {"system": system, **m}
            writer.writerow(row)
    print(f"\nWrote results to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run job-search systems (Steps 2-3) and/or evaluate their "
                    "saved rankings against qrels (Steps 4-6)."
    )
    parser.add_argument("--eval-file", default="eval_dataset.json",
                        help="Path to the labelled qrels file — either a judged-pool CSV "
                             "(query_id,query,job_id,...,relevance,...) or eval_dataset.json. "
                             "Format is auto-detected from the .csv/.json extension. "
                             "(default: eval_dataset.json)")
    parser.add_argument("--rankings-dir", default="rankings",
                        help="Directory to write/read <system>_rankings.csv files (default: rankings/)")
    parser.add_argument("--k", type=int, default=10,
                        help="Cutoff for NDCG/Recall/Precision/MRR/MAP and number of results returned (default: 10)")
    parser.add_argument("--rerank-pool", type=int, default=30,
                        help="Candidate pool size from hybrid search before reranking (default: 30)")
    parser.add_argument("--reps", type=int, default=3,
                        help="Number of repetitions per query for latency measurement during --run (default: 3)")
    parser.add_argument("--output", default=None,
                        help="Optional path to write a CSV summary (used with --evaluate)")
    parser.add_argument("--write-example", action="store_true",
                        help="If --eval-file doesn't exist, write an example dataset there and exit")

    parser.add_argument("--run", action="store_true",
                        help="Step 2/3: run all systems on all queries and save ranked outputs to --rankings-dir")
    parser.add_argument("--evaluate", action="store_true",
                        help="Step 4/5/6: compare saved rankings in --rankings-dir against qrels and report metrics + latency")

    args = parser.parse_args()

    eval_path = Path(args.eval_file)

    if not eval_path.exists():
        if args.write_example:
            write_example_dataset(eval_path)
            return
        print(f"Eval file not found: {eval_path}\n"
              f"Run again with --write-example to generate a template.", file=sys.stderr)
        sys.exit(1)

    if not args.run and not args.evaluate:
        print("Specify --run, --evaluate, or both. See --help.", file=sys.stderr)
        sys.exit(1)

    dataset = load_eval_dataset(eval_path)
    print(f"Loaded {len(dataset)} labelled queries from {eval_path}")

    rankings_dir = Path(args.rankings_dir)
    latencies_by_system = None

    if args.run:
        print(f"\n[Step 2/3] Running systems natively "
              f"(k={args.k}, reps={args.reps}, rerank_pool={args.rerank_pool})...\n")
        latencies_by_system = run_systems_and_save_rankings(
            dataset, k=args.k, rerank_pool=args.rerank_pool,
            reps=args.reps, rankings_dir=rankings_dir,
        )

    if args.evaluate:
        print(f"\n[Step 4/5/6] Evaluating rankings in {rankings_dir} against qrels...\n")
        summary = evaluate_from_rankings(
            dataset, k=args.k, rankings_dir=rankings_dir,
            latencies_by_system=latencies_by_system,
        )

        print()
        print_summary(summary)

        if not latencies_by_system:
            print("\nNote: latency columns are 0 because rankings were loaded from a "
                  "previous --run rather than measured in this invocation. "
                  "Pass --run together with --evaluate to get fresh latency numbers.")

        if args.output:
            write_csv(summary, Path(args.output))


if __name__ == "__main__":
    main()