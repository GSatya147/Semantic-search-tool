#### okay insights from the three chunks

Implemented chunk sizes experiments with 256, 500 and 1024 and overlapping chunk size of 15%.

1. precision queries were not impacted, LLM gave good answers
2. aggregation queries were underperforming as expected cus we are sending reranking results to the LLM
3. chunk_256 was underperforming and gave mid cut off answers by LLM, as expected chunk_500 and chunk_1024 were not that different with each other but chunk 1024 is more useful as per my observation considering the massive voyage context, there is no reason to unnecessarily cut off at 500
**Note:** honestly chunking at token levels is fine but not that adaptive in my case, i'd actually prefer recursive chunking (Chip Huyen, 2025). Fix a max chunk_size and recursively split documents to sections to paragraphs to sentences to words to tokens.. at each component until it hit the max chunk_size, use "\n" as splitters which is naive but works.