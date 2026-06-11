#### Semantic search tool
*stack:* chromadb | voyage-4-nano

#### `chunker.py`
**Class:** `Chunker(chunk_size, file_path)`

- Loads `voyageai/voyage-4-nano` via `SentenceTransformer` with `trust_remote_code=True, truncate_dim=1024`
- Reads file with `encoding="utf-8"`, explicit, platform-safe (Windows defaults to `cp1252`, not UTF-8)
- Tokenizes full document **once** with `model.tokenizer.encode()`
- Chunks via **sliding window**: `tokens[i : i + CHUNK_SIZE]`, step = `CHUNK_SIZE - CHUNK_OVERLAP`
- **Overlap = 15% of chunk size** — prevents information loss at boundaries
- Decodes each token slice back to string with `tokenizer.decode()`
- Returns `list[dict]` — each dict has:
  - `content` — decoded chunk text
  - `metadata` — `source`, `document_name`, `chunk_index`, `token_count`
- `sanity_check()` prints total tokens, chunk count, first/last chunk sizes
- **Verify with mental math:** `(chunks × chunk_size) - (chunks-1 × overlap) = total_tokens`

---

#### `embedder.py`

**Class:** `VectorEmbedder()`

- Loads `voyageai/voyage-4-nano` **once in `__init__`** — never reloads per call
- `EMBED_AND_STORE` flag gates entire operation — **set `False` after first run**
- **Batching:** accumulates chunks until `running_token_count > 5000`, then flushes to `model.encode()` — no half-chunk processing
- Tail batch flushed **after loop** with `if current_batch`
- Uses `list.extend()` not `+=` — flattens 2D numpy output into flat embedding list
- `chromadb.PersistentClient(path="./chromadb")` — survives process restarts
- `get_or_create_collection()` — idempotent, safe to call repeatedly
- `collection.add()` takes **parallel lists** — `ids`, `documents`, `metadatas`, `embeddings` must all be same length

---

#### `retriever.py`

**Class:** `Retreiver()`

- Loads `voyageai/voyage-4-nano` **once in `__init__`** — encodes query immediately on instantiation, never re-encodes per call
- `truncate_dim=1024` — explicit dimensionality lock, matches ChromaDB collection dimensions
- `corpus_retriever(n_results=5, where=None)` — optional metadata filter passed via `where`, conditionally unpacked into `**query_kwargs` — ChromaDB throws on explicit `where=None`
- Filter syntax: MongoDB-style operators — `{"document_name": {"$eq": "main_arcs.txt"}}`, `{"chunk_index": {"$gte": 5}}`
- `PersistentClient` and `get_collection()` called per retrieval — stateless, survives process restarts
- `generate_response(reranker_results)` — accepts reranker output, not raw retriever output — context built from reranked `content` fields with document ID attribution
- System prompt enforces grounded answers — LLM instructed to respond with fallback string if context is inadequate, no hallucination
- LLM model loaded from env via `MODEL` — routed through `litellm.completion()`, swap model without touching code
- Caller owns reranker instantiation — retriever does not know about reranking

---

**Class:** `Reranker()`

- Loads `cross-encoder/ms-marco-MiniLM-L6-v2` **once in `__init__`** — never reloads per call
- **CrossEncoder ≠ SentenceTransformer** — takes `(query, document)` pairs as joint input, outputs raw logits not cosine similarity
- `rerank_results(retriever_result, top_n=3)` — accepts raw ChromaDB result dict directly, owns the transformation
- **Pair construction:** `[query, doc]` per candidate — same query repeated across all pairs, one forward pass per pair
- Scores are **raw logits** — not probabilities, not cosine similarity. Absolute values meaningless, relative ordering is everything. Can be negative.
- Results sorted descending by score, sliced to `top_n` — returns list of dicts with `document_id`, `score`, `content`
- `content` included in return object — caller passes directly to LLM, no second ChromaDB lookup needed
- Retriever instantiation stays in caller — reranker accepts results, does not own retrieval

---