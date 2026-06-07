import os

from sentence_transformers import SentenceTransformer

class chunker:
    def __init__(self, CHUNK_SIZE, CHUNK_OVERLAP, FILE_PATH):
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP
        self.FILE_PATH = FILE_PATH

        with open(self.FILE_PATH, 'r', encoding="utf-8") as f:
            data = f.read()

        try:
            self.model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True, truncate_dim=1024)
            self.tokenizer = self.model.tokenizer

            self.tokenized_data = self.tokenizer.encode(data)
        except Exception as e:
            print(e)

    def corpus_chunker(self)-> list[dict]:
        chunks = []

        i=0
        while i < len(self.tokenized_data):
            chunks.append(self.tokenized_data[i: self.CHUNK_SIZE + i])
            i+=self.CHUNK_SIZE - self.CHUNK_OVERLAP

        document_dict = dict()
        chunk_dict_list = []

        for i in range(len(chunks)):
            document_dict = {
                "content": self.tokenizer.decode(chunks[i]),
                "metadata": {
                    "source": "manual",
                    "document_name": os.path.basename(self.FILE_PATH),
                    "chunk_index": i,
                    "token_count": len(chunks[i])
                }
            }
            chunk_dict_list.append(document_dict)

        return chunk_dict_list

    def sanity_check(self, tokenized_data: list, chunks: list)-> None:
        print(f"Total tokens: {len(tokenized_data)}")
        print(f"Chunks: {len(chunks)}")
        print(f"First chunk length: {len(chunks[0])}")
        print(f"Pen-ultimate chunk length: {len(chunks[-2])}")
        print(f"Ultimate chunk length: {len(chunks[-1])}")   

if __name__=="__main__":
    obj = chunker(500, 75, "C:/Users/gvvsn/OneDrive/Desktop/Learning/Document Corpus/main_arcs.txt")
    result_list = obj.corpus_chunker()
    print(result_list)