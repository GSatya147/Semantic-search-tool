from sentence_transformers.cross_encoder import CrossEncoder

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
        for i in self.retriever_result["documents"][0]:
            batches = [self.user_query, i]
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