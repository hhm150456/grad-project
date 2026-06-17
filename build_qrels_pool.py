"""
build_qrels_pool.py
Implements Steps 1-4 of the qrels-creation workflow:

  Step 1: Queries        -> read from --queries-file (CSV: query_id,query)
  Step 2: Indexed jobs   -> assumed already indexed in Elasticsearch (no-op here)
  Step 3: Retrieve       -> run BM25, Dense (kNN), and Hybrid retrieval for
                             each query, top --pool-depth each (default 20)
  Step 4: Pool + dedupe   -> merge the three result sets per query, drop
                             duplicate job_ids, and write a judgment
                             worksheet (CSV) with one row per (query, job)
                             candidate pair, ready for manual relevance
                             labelling (Step 5).

This script calls elastic_service / embedding_service directly (no HTTP),
matching evaluate_search_native.py, and assumes Elasticsearch + the jobs
index are already populated.

USAGE
-----
    # 1) Generate the judgment worksheet from your 300 queries
    python build_qrels_pool.py \\
        --queries-file queries.csv \\
        --pool-depth 20 \\
        --output pool_to_judge.csv

    # 2) Open pool_to_judge.csv, fill in the empty "relevance" column for
    #    every row (0 = not relevant, 1 = somewhat relevant, 2 = highly
    #    relevant — or just 0/1 for binary judging).

    # 3) Convert your judged worksheet into:
    #      - a TREC-style qrels.txt (query_id 0 job_id relevance)
    #      - eval_dataset.json, ready for evaluate_search_native.py
    python build_qrels_pool.py --convert pool_to_judge.csv \\
        --qrels-out qrels.txt \\
        --eval-json-out eval_dataset.json

INPUT: queries.csv
-------------------
    query_id,query
    Q1,Python backend developer with Django
    Q2,Entry-level data analyst in London
    Q3,Remote React frontend engineer

OUTPUT: pool_to_judge.csv (Step 4 deliverable — hand this to your judges)
--------------------------------------------------------------------------
    query_id,query,job_id,job_title,job_description,sources,relevance
    Q1,Python backend developer with Django,J15,Python Django Developer,...,bm25;dense,
    Q1,Python backend developer with Django,J28,Senior Backend Engineer...,...,bm25;hybrid,
    ...

"sources" records which retrieval method(s) surfaced that job (e.g.
"bm25;dense;hybrid"), purely for your own diagnostics — it is not used by
the evaluation metrics, all candidates are judged equally.

Leave "relevance" blank, fill it in (0/1, or 0/1/2 for graded relevance)
during Step 5, then run --convert to produce qrels.

OUTPUT: qrels.txt (TREC format)
---------------------------------
    Q1 0 J15 1
    Q1 0 J28 1
    Q1 0 J42 0

OUTPUT: eval_dataset.json (consumed by evaluate_search_native.py)
---------------------------------------------------------------------
    [
      {"query": "Python backend developer with Django",
       "relevant": {"J15": 1, "J28": 1, "J42": 0}},
      ...
    ]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))



def load_queries(path: Path) -> list[dict]:
    """
    Reads a CSV with columns: query_id, query
    Returns a list of {"query_id": ..., "query": ...} dicts, in file order.
    """
    queries = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "query_id" not in reader.fieldnames or "query" not in reader.fieldnames:
            raise ValueError(
                f"{path} must have 'query_id' and 'query' columns, "
                f"got columns: {reader.fieldnames}"
            )
        for row in reader:
            qid = row["query_id"].strip()
            q = row["query"].strip()
            if qid and q:
                queries.append({"query_id": qid, "query": q})
    return queries


def write_example_queries(path: Path) -> None:
    rows = [
        {"query_id": "Q1", "query": "Python backend developer with Django"},
        {"query_id": "Q2", "query": "Entry-level data analyst in London"},
        {"query_id": "Q3", "query": "Remote React frontend engineer"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote example queries file to {path}.")




def retrieve_bm25(query: str, pool_depth: int) -> list[dict]:
    from elastic_service import bm25_search_jobs
    return bm25_search_jobs(query=query, k=pool_depth)


def retrieve_dense(query: str, pool_depth: int) -> list[dict]:
    from elastic_service import knn_search_jobs
    from embedding_service import generate_query_embedding
    embedding = generate_query_embedding(query, seeker_embedding=None)
    return knn_search_jobs(query_embedding=embedding, k=pool_depth,
                            num_candidates=max(100, pool_depth * 10))


def retrieve_hybrid(query: str, pool_depth: int) -> list[dict]:
    from elastic_service import hybrid_search_jobs
    from embedding_service import generate_query_embedding
    embedding = generate_query_embedding(query, seeker_embedding=None)
    return hybrid_search_jobs(query=query, query_embedding=embedding, k=pool_depth)


RETRIEVERS = {
    "bm25": retrieve_bm25,
    "dense": retrieve_dense,
    "hybrid": retrieve_hybrid,
}




def build_pool_for_query(query: str, pool_depth: int) -> list[dict]:
    
    pool: dict[str, dict] = {}

    for source_name, retriever in RETRIEVERS.items():
        results = retriever(query, pool_depth)
        for r in results:
            job_id = r["job_id"]
            if job_id not in pool:
                pool[job_id] = {
                    "job_id": job_id,
                    "job_title": r["job_title"],
                    "job_description": r["job_description"],
                    "sources": [],
                }
            if source_name not in pool[job_id]["sources"]:
                pool[job_id]["sources"].append(source_name)

    return list(pool.values())


def build_full_pool(queries: list[dict], pool_depth: int) -> list[dict]:
    """
    Returns a flat list of rows, one per (query, candidate job) pair, ready
    to write to the judgment worksheet CSV.
    """
    rows = []
    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]
        print(f"  Pooling {query_id}: {query_text!r}")

        candidates = build_pool_for_query(query_text, pool_depth)
        print(f"    -> {len(candidates)} unique candidates after dedup")

        for c in candidates:
            rows.append({
                "query_id": query_id,
                "query": query_text,
                "job_id": c["job_id"],
                "job_title": c["job_title"],
                "job_description": c["job_description"],
                "sources": ";".join(c["sources"]),
                "relevance": "",
            })

    return rows


def write_judgment_worksheet(rows: list[dict], path: Path) -> None:
    fieldnames = ["query_id", "query", "job_id", "job_title",
                  "job_description", "sources", "relevance"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote judgment worksheet ({len(rows)} rows) to {path}")
    print("Next: open this file and fill in the 'relevance' column for every row "
          "(0 = not relevant, 1 = relevant, or 0/1/2 for graded relevance), "
          "then run with --convert to produce qrels.")




def convert_judged_worksheet(path: Path, qrels_out: Path, eval_json_out: Path) -> None:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "query", "job_id", "relevance"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        rows = list(reader)

    unjudged = [r for r in rows if r["relevance"].strip() == ""]
    if unjudged:
        print(f"WARNING: {len(unjudged)} row(s) still have an empty 'relevance' value. "
              f"They will be skipped. First few: "
              f"{[(r['query_id'], r['job_id']) for r in unjudged[:5]]}", file=sys.stderr)

    # Group by query, preserving first-seen query order and query text.
    by_query_order: list[str] = []
    query_text_by_id: dict[str, str] = {}
    relevant_by_query: dict[str, dict[str, int]] = {}

    for row in rows:
        rel_str = row["relevance"].strip()
        if rel_str == "":
            continue
        try:
            rel = int(rel_str)
        except ValueError:
            print(f"WARNING: non-integer relevance {rel_str!r} for "
                  f"{row['query_id']}/{row['job_id']}, skipping.", file=sys.stderr)
            continue

        qid = row["query_id"]
        if qid not in relevant_by_query:
            relevant_by_query[qid] = {}
            query_text_by_id[qid] = row["query"]
            by_query_order.append(qid)

        relevant_by_query[qid][row["job_id"]] = rel

    # qrels.txt — TREC format: query_id 0 job_id relevance
    with open(qrels_out, "w", encoding="utf-8") as f:
        for qid in by_query_order:
            for job_id, rel in relevant_by_query[qid].items():
                f.write(f"{qid} 0 {job_id} {rel}\n")
    print(f"Wrote {qrels_out}")

    # eval_dataset.json — consumed directly by evaluate_search_native.py
    eval_dataset = [
        {"query": query_text_by_id[qid], "relevant": relevant_by_query[qid]}
        for qid in by_query_order
    ]
    with open(eval_json_out, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=2)
    print(f"Wrote {eval_json_out} ({len(eval_dataset)} queries) — "
          f"ready for: python evaluate_search_native.py --eval-file {eval_json_out}")



def main():
    parser = argparse.ArgumentParser(
        description="Build a pooled candidate set (Steps 3-4) for qrels judging, "
                    "and convert a judged worksheet into qrels.txt / eval_dataset.json."
    )
    parser.add_argument("--queries-file", default="queries.csv",
                        help="CSV with columns query_id,query (default: queries.csv)")
    parser.add_argument("--pool-depth", type=int, default=20,
                        help="Top-N retrieved per method before pooling/dedup (default: 20)")
    parser.add_argument("--output", default="pool_to_judge.csv",
                        help="Output judgment worksheet CSV (default: pool_to_judge.csv)")
    parser.add_argument("--write-example-queries", action="store_true",
                        help="If --queries-file doesn't exist, write an example file there and exit")

    parser.add_argument("--convert", default=None, metavar="JUDGED_CSV",
                        help="Path to a judged worksheet CSV (with 'relevance' filled in). "
                            "If given, skips pooling and instead converts this file into "
                            "qrels + eval_dataset.json.")
    parser.add_argument("--qrels-out", default="qrels.txt",
                        help="Output path for TREC-format qrels (used with --convert)")
    parser.add_argument("--eval-json-out", default="eval_dataset.json",
                        help="Output path for eval_dataset.json (used with --convert)")

    args = parser.parse_args()

    if args.convert:
        convert_judged_worksheet(
            Path(args.convert),
            Path(args.qrels_out),
            Path(args.eval_json_out),
        )
        return

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        if args.write_example_queries:
            write_example_queries(queries_path)
            return
        print(f"Queries file not found: {queries_path}\n"
              f"Run again with --write-example-queries to generate a template.",
              file=sys.stderr)
        sys.exit(1)

    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries from {queries_path}")
    print(f"Pooling top-{args.pool_depth} from bm25 + dense + hybrid for each query...\n")

    rows = build_full_pool(queries, pool_depth=args.pool_depth)
    write_judgment_worksheet(rows, Path(args.output))


if __name__ == "__main__":
    main()
