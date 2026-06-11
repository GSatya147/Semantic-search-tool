from sentence_transformers.cross_encoder import CrossEncoder
import chromadb

from retriever import Retriever

class Reranker:
    def __init__(self, user_query):
        self.user_query = user_query

        try:
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        except Exception as e:
            print(e)

    def rerank_results(self):
        retreiver_obj = Retriever(self.user_query)
        self.retriever_result = retreiver_obj.corpus_retriever()

        self.sentences = []
        for result in self.retriever_result:
            batches = [self.user_query, result]
            self.sentences.append(batches)

        self.scores = self.model.predict(self.sentences)
        
        return self.scores, self.retriever_result, self.sentences

if __name__=="__main__":
    user_query = input(">> ")
    obj = Reranker(user_query)
    reranker_result, retriever_result, sentences = obj.rerank_results()
    print(f"sentences: {sentences}")
    print(f"reranker: {reranker_result}")
    print(f"retriever: {retriever_result}")