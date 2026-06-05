from sentence_transformers import CrossEncoder
from src.config import Config

class Reranker:
    def __init__(self):
        self.model = None

    def load_model(self, model_name: str = Config.RERANK_MODEL_NAME):
        """
        Loads the cross-encoder reranker model.
        """
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Reranker model loaded successfully.")

    def rerank(self, query: str, doc_list: list, corpus: dict, top_k: int = Config.TOP_K_RERANK) -> list:
        """
        Reranks a list of candidate documents against a query using the Cross-Encoder.
        
        Args:
            query: The search query string.
            doc_list: A list of dicts: [{'corpus-id': doc_id, 'score': score}] representing candidates.
            corpus: A dictionary mapping 'corpus-id' to the raw document text.
            top_k: The number of top candidates to return after reranking.
            
        Returns:
            A reranked list of dicts: [{'corpus-id': doc_id, 'score': rerank_score}] sorted in descending order.
        """
        if self.model is None:
            raise ValueError("Reranker model is not loaded. Call load_model() first.")
            
        if not doc_list:
            return []
            
        doc_ids = [item["corpus-id"] for item in doc_list]
        
        # Build query-document pairs
        pairs = []
        for doc_id in doc_ids:
            doc_text = corpus.get(doc_id, "")
            pairs.append([query, doc_text])
            
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Zip, sort, and return top_k
        results = []
        for doc_id, score in zip(doc_ids, scores):
            results.append({
                "corpus-id": doc_id,
                "score": float(score)
            })
            
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
