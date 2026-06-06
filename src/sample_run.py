import json
import os
import sys
import argparse

# Reconfigure encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.config import Config
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.fusion import reciprocal_rank_fusion
from src.reranker import Reranker
from src.generator import LLMGenerator

def main():
    parser = argparse.ArgumentParser(description="Interactive pipeline demo on mini sample")
    parser.add_argument(
        "--build-indices",
        action="store_true",
        help="Force rebuild of indices (default: load if exists)"
    )
    args = parser.parse_args()

    print("="*60)
    print("INTERACTIVE PIPELINE DEMO ON MINI SAMPLE")
    print("Type 'exit' or 'quit' to stop")
    print("="*60)

    # 1. Create a mini corpus of 100 documents for local testing
    mini_corpus_path = "src/Data/processed/mini_corpus.jsonl"
    print(f"Creating mini corpus at {mini_corpus_path}...")

    docs = []
    with open(Config.CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            docs.append(json.loads(line))

    with open(mini_corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    corpus_map = {doc["_id"]: doc["text"] for doc in docs}
    print(f"Mini corpus created with {len(docs)} documents.")

    # Initialize components
    print("\nInitializing components...")

    # 2. BM25 Retriever
    mini_bm25_path = "src/Data/processed/mini_bm25_index.pkl"
    bm25 = BM25Retriever()
    if os.path.exists(mini_bm25_path) and not args.build_indices:
        print("Loading existing BM25 index...")
        bm25.load_index(mini_bm25_path)
    else:
        print("Building BM25 index...")
        bm25.build_index(mini_corpus_path, mini_bm25_path)

    # 3. Dense Retriever
    mini_dense_path = "src/Data/processed/mini_dense_index.faiss"
    mini_doc_ids_path = "src/Data/processed/mini_dense_doc_ids.json"
    dense = DenseRetriever()
    if os.path.exists(mini_dense_path) and os.path.exists(mini_doc_ids_path) and not args.build_indices:
        print("Loading existing Dense index...")
        dense.load_index(mini_dense_path, mini_doc_ids_path)
    else:
        print("Building Dense index...")
        dense.build_index(mini_corpus_path, mini_dense_path, mini_doc_ids_path)

    # 4. Reranker
    reranker = Reranker()
    print("Loading reranker model...")
    reranker.load_model(Config.RERANK_MODEL_NAME)

    # 5. LLM Generator
    generator = LLMGenerator()
    print("Loading LLM generator model...")
    generator.load_model(Config.LLM_MODEL_NAME)

    print("\n" + "="*60)
    print("All components ready! Enter your queries below.")
    print("="*60)

    # Interactive query loop
    while True:
        try:
            # Get query from user
            query = input("\nNhập câu hỏi của bạn (hoặc 'exit'/'quit' để thoát): ").strip()

            # Check for exit condition
            if query.lower() in ['exit', 'quit', '']:
                print("\nCảm bạn đã sử dụng hệ thống!")
                break

            print(f"\nProcessing query: '{query}'")

            print("\n[Step 1] Lexical Retrieval (BM25)...")
            bm25_res = bm25.retrieve(query, top_k=10)
            for r in bm25_res[:3]:
                print(f"  Doc ID: {r['corpus-id']} | Score: {r['score']:.4f}")

            print("\n[Step 2] Dense Vector Retrieval...")
            dense_res = dense.retrieve(query, top_k=10)
            for r in dense_res[:3]:
                print(f"  Doc ID: {r['corpus-id']} | Score: {r['score']:.4f}")

            print("\n[Step 3] Reciprocal Rank Fusion (RRF)...")
            fused_res = reciprocal_rank_fusion([bm25_res, dense_res])
            for r in fused_res[:3]:
                print(f"  Doc ID: {r['corpus-id']} | Fused Score: {r['score']:.6f}")

            print("\n[Step 4] Cross-Encoder Reranking...")
            reranked_res = reranker.rerank(
                query=query,
                doc_list=fused_res[:10],
                corpus=corpus_map,
                top_k=3
            )
            for r in reranked_res:
                print(f"  Doc ID: {r['corpus-id']} | Reranked Score: {r['score']:.4f}")

            # 6. Generate answer with LLM (full RAG)
            print("\n[Step 5] Generating answer with LLM...")
            answer = generator.generate(query, reranked_res, corpus_map)
            print("\n" + "="*60)
            print(f"Câu hỏi: {query}")
            print(f"Câu trả lời: {answer}")
            print("="*60)

        except KeyboardInterrupt:
            print("\n\nNhận tín hiệu dừng. Tạm biệt!")
            break
        except Exception as e:
            print(f"\nLỗi xảy ra: {e}")
            print("Vui lòng thử lại với câu hỏi khác.")

    # Clean up temporary mini files
    print("\nCleaning up temporary test files...")
    for path in [mini_corpus_path, mini_bm25_path, mini_dense_path, mini_doc_ids_path]:
        if os.path.exists(path):
            os.remove(path)

    print("\nDEMO ENDED!")
    print("="*60)

if __name__ == "__main__":
    main()