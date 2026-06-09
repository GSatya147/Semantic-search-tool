import chromadb
from sentence_transformers import SentenceTransformer
from groq import model

class Retriever:
    def __init__(self, user_query):
        self.query = user_query

        try:
            self.model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True ,truncate_dim=1024)
            self.query_embeddings = self.model.encode(self.query)

        except Exception as e:
            print(e)

    def corpus_retriever(self):
        client = chromadb.PersistentClient(path="./chromadb")

        collections = client.get_collection(name="one_piece_main_arcs")

        self.result = collections.query(
            query_embeddings=self.query_embeddings,
            n_results=3
        )

        return self.result
    
    def generate_response(self):
        self.corpus_retriever()

        context = "/n/n".join()

        sys_prompt = f"""
            you are an one piece knowledge assistant, answer only on the basis of provided context:
            <context>{context}</context>
            if the given context is inadequate, just answer "It is out of the provided knowledge"
        """


        


if __name__=="__main__":
    query = input(">> ")

    obj = Retriever(query)
    obj.corpus_retriever()

    print(f"ids: {obj.result["ids"]}")
    print(f"documents: {obj.result['documents']}")
    print(f"distances: {obj.result["distances"]}")