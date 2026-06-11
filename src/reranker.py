from sentence_transformers.cross_encoder import CrossEncoder
import chromadb

from retriever import Retriever

class Reranker:
    def __init__(self, user_query):
        self.user_query = user_query

        try:
            self.model = CrossEncoder("")
        except Exception as e:
            print(e)

    def rerank_results(self):
        retreiver_obj = Retriever(self.user_query)
        retriever_result = retreiver_obj.corpus_retriever()

        self.sentences = []
        for result in retriever_result:
            batches = [self.user_query, result]
            self.sentences.append(batches)

        self.scores = self.model.predict(self.sentences)
        
        return self.scores

if __name__=="__main__":
    user_query = input(">> ")
    obj = Reranker(user_query)
    result = obj.rerank_results()
    print(result)