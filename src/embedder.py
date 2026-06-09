import chromadb
from chunker import Chunker
from sentence_transformers import SentenceTransformer

class VectorEmbedder:
    def __init__(self):
        self.EMBED_AND_STORE = False

        try:
            self.model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True, truncate_dim=1024)
        except Exception as e:
            print(e)

    def corpus_embedder(self, chunk_size, file_path):
        chunker_obj = Chunker(chunk_size, file_path, self.model)
        self.chunker_result = chunker_obj.corpus_chunker()

        try:
            self.embeddings = []
            running_tokens = 0
            current_batch = []
            for chunk in self.chunker_result:
                if running_tokens + chunk["metadata"]["token_count"] > 5000: 
                    self.embeddings.extend(self.model.encode(current_batch))

                    running_tokens = 0
                    current_batch = []
                
                running_tokens+=chunk["metadata"]["token_count"]
                current_batch.append(chunk["content"])
            
            if current_batch:
                self.embeddings.extend(self.model.encode(current_batch))

        except Exception as e:
            print(e)
        
        return self.embeddings
    
    def vector_store(self, chunk_size, filepath):
        try:
            if self.EMBED_AND_STORE:
                self.corpus_embedder(chunk_size, filepath)

                client = chromadb.PersistentClient(path="./chromadb")

                collection = client.get_or_create_collection(name="one_piece_main_arcs", metadata={"description": "all main story arcs in one piece present manga"})
                
                collection.add(
                    ids= [f"doc_{i}" for i in range(len(self.embeddings))],
                    documents= [chunk["content"] for chunk in self.chunker_result],
                    metadatas= [chunk["metadata"] for chunk in self.chunker_result],
                    embeddings= self.embeddings,
                )

        except Exception as e:
            print(e)

if __name__=="__main__":
    obj = VectorEmbedder()
    obj.vector_store(500, "./main_arcs.txt")
    if obj.EMBED_AND_STORE:
        print(obj.embeddings)
        print(len(obj.embeddings))