import os

class Config:
    # Data Paths
    BASE_DIR = "src/Data/processed"
    CORPUS_PATH = os.path.join(BASE_DIR, "corpus.jsonl")
    QUERIES_PATH = os.path.join(BASE_DIR, "queries.jsonl")
    TRAIN_QRELS_PATH = os.path.join(BASE_DIR, "qrels/train.jsonl")
    TEST_QRELS_PATH = os.path.join(BASE_DIR, "qrels/test.jsonl")

    # Index Saving Paths
    BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")
    DENSE_INDEX_PATH = os.path.join(BASE_DIR, "dense_index.faiss")
    DENSE_DOC_IDS_PATH = os.path.join(BASE_DIR, "dense_doc_ids.json")

    # Models Configuration
    # BGE-M3: public multilingual embedding model (multi-lingual + multi-functionality)
    DENSE_MODEL_NAME = "BAAI/bge-m3"
    RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
    LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    # Search & Fusion parameters
    TOP_K_RETRIEVE = 50     # Candidates from BM25 and Dense each
    TOP_K_RRF = 20          # Candidates after RRF fusion
    TOP_K_RERANK = 5        # Final candidates passed to LLM after Cross-Encoder reranking
    
    # BM25 parameters
    BM25_K1 = 1.5
    BM25_B = 0.75
    
    # RRF parameter
    RRF_K = 60

    # LLM parameters
    LLM_MAX_NEW_TOKENS = 300
    LLM_TEMPERATURE = 0.1
