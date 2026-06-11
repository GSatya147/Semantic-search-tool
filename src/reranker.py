from sentence_transformers.cross_encoder import CrossEncoder

from retriever import Retriever

class Reranker:
    def __init__(self, user_query):
        self.user_query = user_query

        try:
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        except Exception as e:
            print(e)

    def rerank_results(self, documents, top_n=3):
        sentences = []
        for i in documents:
            batches = [self.user_query, documents]
            sentences.append(batches)

        self.scores = self.model.predict(sentences)
        
        ranking_list = []
        for i, j in zip(self.scores, documents):
            ranking_list.append({"document" : j, "score" : i})
        
        self.reranker_results = sorted(ranking_list, key=lambda x: x["score"], reverse=True)

        return self.reranker_results[:top_n]

if __name__=="__main__":
    user_query = input(">> ")
    obj = Reranker(user_query)
    retreiver_obj = Retriever(user_query)
    retriever_result = retreiver_obj.corpus_retriever()

    documents = retriever_result["documents"][0]
    reranker_result,sentences = obj.rerank_results(documents=documents)

    print(f"sentences: {sentences}")
    print(f"reranker: {reranker_result}")
    print(f"retriever: {retriever_result}")