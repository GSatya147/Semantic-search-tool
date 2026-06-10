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

#### `retriever.py`

**Class:** `Retreiver()`