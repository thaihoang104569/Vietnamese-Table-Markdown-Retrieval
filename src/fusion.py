from src.config import Config

def reciprocal_rank_fusion(retrieval_results_list: list, k: int = Config.RRF_K) -> list:
    """
    Combines multiple retrieval lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        retrieval_results_list: A list where each element is a list of retrieval results
                                (dicts containing 'corpus-id' and 'score').
        k: The constant parameter for the RRF formula (default: 60).
        
    Returns:
        A combined and re-ranked list of dicts: [{'corpus-id': doc_id, 'score': fused_score}]
        sorted by score in descending order.
    """
    fused_scores = {}
    for results in retrieval_results_list:
        for rank, item in enumerate(results):
            doc_id = item["corpus-id"]
            # rank is 0-indexed, so rank+1 is the 1-based rank
            rank_score = 1.0 / (k + (rank + 1))
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rank_score
            
    # Convert dict to sorted list of results
    sorted_results = sorted(
        [{"corpus-id": doc_id, "score": score} for doc_id, score in fused_scores.items()],
        key=lambda x: x["score"],
        reverse=True
    )
    return sorted_results
