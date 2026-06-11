import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from litellm import completion

load_dotenv()

class Retriever:
    def __init__(self, user_query):
        self.query = user_query

        try:
            self.model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True ,truncate_dim=1024)
            self.query_embeddings = self.model.encode(self.query)

        except Exception as e:
            print(e)

    def corpus_retriever(self, n_results=5, where=None):
        client = chromadb.PersistentClient(path="./chromadb")

        collections = client.get_collection(name="one_piece_main_arcs")

        query_qwargs = {
            "query_embeddings": self.query_embeddings,
            "n_results" : n_results,
        }
        if where is not None:
            query_qwargs["where"] = where

        self.result = collections.query(**query_qwargs)

        return self.result
    
    def generate_response(self, reranker_results):
        
        context = "\n\n".join([
            f"[{reranker_results[i]["document_ids"]}]: {reranker_results[i]["content"]}"
            for i in range(len(reranker_results))
        ])

        sys_prompt = f"""
            you are an one piece knowledge assistant, answer only on the basis of provided context:
            <context>{context}</context>
            if the given context is inadequate, just answer "It is out of the provided knowledge" only.
        """

        try:
            self.response = completion(
                model=os.getenv("MODEL"),
                messages=[
                    {"role" : "system", "content" : sys_prompt},
                    {"role" : "user", "content": self.query},
                ]
            )

            return self.response
        except Exception as e:
            print(e)

if __name__=="__main__":
    query = input(">> ")

    obj = Retriever(query)
    result = obj.generate_response()

    print(result.choices[0].message.content)