import os

import chromadb
from src.chunker import Chunker
from sentence_transformers import SentenceTransformer

class VectorEmbedder:
    def __init__(self,):
        pass

    def corpus_embedder(self, chunk_size, file_path):
        chunker_obj = Chunker(chunk_size, file_path)
        chunker_result = chunker_obj.corpus_chunker()

        try:
            model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True, truncate_dim=1074)

            self.embeddings = []
            running_tokens = 0
            current_batch = []
            for chunk in chunker_result:
                if running_tokens + chunk["metadata"]["token_count"] > 5000: 
                    self.embeddings+=model.encode(current_batch)

                    running_tokens = 0
                    current_batch = []
                
                running_tokens+=chunk["metadata"]["token_count"]
                current_batch.append(chunk)
            
            if current_batch:
                result = model.encode(current_batch)

        except Exception as e:
            print(e)
        
        return self.embeddings
    
    def vector_store(self):
        pass

if __name__=="__main__":
    pass