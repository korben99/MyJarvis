#!/opt/jarvis/venv/bin/python3
"""
Search or inspect the Jarvis knowledge base in Qdrant.

Usage:
  python3 search-qdrant.py "ma question"          # semantic search
  python3 search-qdrant.py --list                 # list all indexed files
  python3 search-qdrant.py --list --filter perso  # filter filenames by keyword
  python3 search-qdrant.py "query" --top 10       # more results
  python3 search-qdrant.py "query" --threshold 0.3  # lower score threshold
"""

import sys, os, json, argparse
from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest

QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION  = os.getenv("COLLECTION", "open-webui_knowledge")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_source(payload: dict) -> str:
    meta = payload.get("metadata") or {}
    return (
        meta.get("name") or
        meta.get("source") or
        payload.get("file_name") or
        "unknown"
    )


def cmd_list(args):
    client = QdrantClient(url=QDRANT_URL)
    files = {}   # source → chunk count
    offset = None

    while True:
        result = client.scroll(
            collection_name=COLLECTION,
            limit=500,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for point in result[0]:
            src = get_source(point.payload)
            files[src] = files.get(src, 0) + 1
        offset = result[1]
        if offset is None:
            break

    filtered = {k: v for k, v in files.items()
                if not args.filter or args.filter.lower() in k.lower()}
    sorted_files = sorted(filtered.items(), key=lambda x: x[0].lower())

    print(f"{'FILE':<70} CHUNKS")
    print("─" * 78)
    for name, count in sorted_files:
        print(f"{name:<70} {count}")
    print(f"\nTotal: {len(sorted_files)} files ({sum(filtered.values())} chunks)"
          + (f" matching '{args.filter}'" if args.filter else ""))


def cmd_search(args):
    from sentence_transformers import SentenceTransformer

    query = args.query
    print(f'Searching: "{query}"\n')

    model = SentenceTransformer(EMBED_MODEL)
    vector = model.encode(query).tolist()

    client = QdrantClient(url=QDRANT_URL)
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=args.top,
        score_threshold=args.threshold,
    )

    if not results.points:
        print(f"No results above score threshold {args.threshold}.")
        print("Try lowering --threshold (current default: 0.4)")
        return

    for i, hit in enumerate(results.points):
        payload = hit.payload
        meta = payload.get("metadata") or {}
        text = payload.get("text", "")
        source = get_source(payload)
        start = meta.get("start_index", "?")

        print(f"── Result {i+1} (score: {hit.score:.3f}) ──")
        print(f"Source : {source}  [offset {start}]")
        print(text[:500])
        print()


def main():
    p = argparse.ArgumentParser(description="Inspect Jarvis Qdrant knowledge base")
    sub = p.add_subparsers(dest="cmd")

    ls = sub.add_parser("--list", help="List indexed files")
    ls.add_argument("--filter", default="", help="Filter filenames by keyword")

    p.add_argument("query", nargs="?", help="Search query")
    p.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument("--threshold", type=float, default=0.4, help="Min score (default: 0.4)")
    p.add_argument("--list", action="store_true", help="List all indexed files")
    p.add_argument("--filter", default="", help="Filter file list by keyword (with --list)")

    args = p.parse_args()

    if args.list:
        cmd_list(args)
    elif args.query:
        cmd_search(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
