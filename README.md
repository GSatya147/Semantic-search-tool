#### Semantic search tool
*stack:* chromadb | voyage-4-nano
#### Chunker
- sliding window of N tokens with M number of overlapping tokens
- M is generally 10-20% of N
- N = chunk size, M = overlapping tokens size
#### Embedder
* Retriever