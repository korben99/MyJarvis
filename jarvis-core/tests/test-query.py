#!/opt/jarvis/venv/bin/python3
"""Verify that vectors in Qdrant are actually searchable, bypassing OpenWebUI."""

import sys
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

QDRANT_URL = "http://localhost:6333"
COLLECTION = "open-webui_knowledge"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

query = " ".join(sys.argv[1:]) or "test"

client = QdrantClient(url=QDRANT_URL)
embed  = HuggingFaceEmbedding(model_name=EMBED_MODEL)

vector  = embed.get_text_embedding(query)
results = client.query_points(collection_name=COLLECTION, query=vector, limit=3).points

if not results:
    print("No results — data may not be indexed correctly.")
else:
    for r in results:
        print(f"Score: {r.score:.4f}")
        print(f"File:  {r.payload.get('file_name') or r.payload.get('metadata', {}).get('name', 'N/A')}")
        print(f"Text:  {(r.payload.get('content') or '')[:200]}")
        print()
