import json
import os
import sys
import math
import argparse
from tqdm import tqdm
from src.config import Config
from src.pipeline import RAGPipeline

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def calculate_metrics(retrieved_ids, target_ids, k_list=[1, 5, 10]):
    """
    Computes Recall@K, MAP@K, and NDCG@K for a single query.
    Target IDs is a set of relevant document IDs.
    Retrieved IDs is a list of doc IDs sorted by score descending.
    """
    metrics = {}
    
    # Pre-calculate intersections and relevance for speed
    rel_hits = [1 if doc_id in target_ids else 0 for doc_id in retrieved_ids]
    
    # 1. Recall@K
    for k in k_list:
        hits_at_k = sum(rel_hits[:k])
        metrics[f"Recall@{k}"] = hits_at_k / len(target_ids) if len(target_ids) > 0 else 0.0

    # 2. MAP@10 (equivalent to Average Precision@10)
    ap = 0.0
    num_hits = 0
    for i, hit in enumerate(rel_hits[:10]):
        if hit == 1:
            num_hits += 1
            ap += num_hits / (i + 1)
    metrics["MAP@10"] = ap / len(target_ids) if len(target_ids) > 0 else 0.0

    # 3. NDCG@10
    dcg = 0.0
    for i, hit in enumerate(rel_hits[:10]):
        if hit == 1:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(10, len(target_ids))):
        idcg += 1.0 / math.log2(i + 2)
        
    metrics["NDCG@10"] = dcg / idcg if idcg > 0.0 else 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Retrieval & RAG Pipeline")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Run RAG pipeline with local LLM generator (requires CUDA GPU and 16GB VRAM)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of test queries to evaluate (run on subset for speed)"
    )
    args = parser.parse_args()

    # Load test queries
    print(f"Loading queries from {Config.QUERIES_PATH}...")
    queries = {}
    with open(Config.QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]
            
    # Load test qrels
    print(f"Loading test qrels from {Config.TEST_QRELS_PATH}...")
    qrels = {}
    with open(Config.TEST_QRELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q_id = item["query-id"]
            d_id = item["corpus-id"]
            qrels.setdefault(q_id, set()).add(d_id)
            
    print(f"Loaded {len(queries)} queries and {len(qrels)} qrels mappings.")

    # Initialize pipeline
    pipeline = RAGPipeline(include_llm=args.with_llm)
    pipeline.initialize()

    # Select queries that have qrels mappings
    eval_q_ids = [q_id for q_id in queries.keys() if q_id in qrels]
    
    # Cap by sample size to save time
    if args.sample_size > 0 and len(eval_q_ids) > args.sample_size:
        print(f"Sampling {args.sample_size} queries for evaluation...")
        eval_q_ids = eval_q_ids[:args.sample_size]
        
    # Accumulate metrics
    accumulated_metrics = {
        "Recall@1": 0.0,
        "Recall@5": 0.0,
        "Recall@10": 0.0,
        "MAP@10": 0.0,
        "NDCG@10": 0.0
    }
    
    count = 0
    
    print("\nEvaluating retrieval pipeline...")
    for q_id in tqdm(eval_q_ids, desc="Evaluating"):
        query_text = queries[q_id]
        target_ids = qrels[q_id]
        
        # Retrieve candidates
        retrieved_candidates = pipeline.retrieve_only(query_text)
        retrieved_ids = [doc["corpus-id"] for doc in retrieved_candidates]
        
        # Compute metrics
        query_metrics = calculate_metrics(retrieved_ids, target_ids)
        for metric, score in query_metrics.items():
            accumulated_metrics[metric] += score
            
        count += 1
        
        # If LLM is active, run RAG on first 3 samples and display
        if args.with_llm and count <= 3:
            print("\n" + "="*50)
            print(f"QUERY: {query_text}")
            rag_output = pipeline.query(query_text)
            print(f"ANSWER:\n{rag_output['answer']}")
            print(f"TOP GROUND TRUTH DOC ID: {list(target_ids)[0]}")
            print(f"RETRIEVED DOC ID: {retrieved_ids[0] if retrieved_ids else 'None'}")
            print("="*50 + "\n")

    # Calculate mean metrics
    print("\n" + "="*30 + " EVALUATION RESULTS " + "="*30)
    print(f"Evaluated on {count} queries.")
    for metric, score in accumulated_metrics.items():
        mean_score = score / count
        print(f"  {metric}: {mean_score:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
