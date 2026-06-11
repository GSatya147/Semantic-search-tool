from sentence_transformers.cross_encoder import CrossEncoder

from retriever import Retriever

class Reranker:
    def __init__(self, user_query):
        self.user_query = user_query

        try:
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        except Exception as e:
            print(e)

    def rerank_results(self, retriever_result, top_n=3):
        documents = retriever_result["documents"][0]
        ids = ids=retriever_result["ids"][0]

        sentences = []
        for i in documents:
            batches = [self.user_query, i]
            sentences.append(batches)

        result = self.model.predict(sentences)
        self.scores = result.tolist()

        ranking_list = []
        for id, score, doc in zip(ids, self.scores, documents):
            ranking_list.extend([{"document_ids" : id, "score" : score, "content" : doc}])
        
        self.reranker_results = sorted(ranking_list, key=lambda x: x["score"], reverse=True)
        return self.reranker_results[:top_n]

if __name__=="__main__":
    user_query = input(">> ")
    obj = Reranker(user_query)
    retriever_obj = Retriever(user_query)
    retriever_results = retriever_obj.corpus_retriever(where={"document_name": {"$eq": "main_arcs.txt"}})

    reranker_result = obj.rerank_results(retriever_result=retriever_results)

    # print(f"reranker: {reranker_result}")

    response = retriever_obj.generate_response(reranker_results=reranker_result)
    print(response.choices[0].message.content)