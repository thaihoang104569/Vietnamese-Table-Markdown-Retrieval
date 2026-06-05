import os
import json
from src.config import Config
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.fusion import reciprocal_rank_fusion
from src.reranker import Reranker
from src.generator import LLMGenerator

class RAGPipeline:
    def __init__(self, include_llm: bool = True):
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.reranker = Reranker()
        self.generator = LLMGenerator() if include_llm else None
        self.corpus = {} # Maps doc_id to raw text

    def initialize(self):
        """
        Loads the indices from disk if they exist, or builds them if not.
        Also loads the reranker model.
        """
        # Load corpus to memory (quick lookup)
        print(f"Loading corpus map from {Config.CORPUS_PATH}...")
        with open(Config.CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.corpus[item["_id"]] = item["text"]
        print(f"Corpus map loaded with {len(self.corpus)} items.")

        # 1. Initialize BM25
        if os.path.exists(Config.BM25_INDEX_PATH):
            self.bm25_retriever.load_index(Config.BM25_INDEX_PATH)
        else:
            self.bm25_retriever.build_index(Config.CORPUS_PATH, Config.BM25_INDEX_PATH)

        # 2. Initialize Dense Retriever
        if os.path.exists(Config.DENSE_INDEX_PATH) and os.path.exists(Config.DENSE_DOC_IDS_PATH):
            self.dense_retriever.load_index(Config.DENSE_INDEX_PATH, Config.DENSE_DOC_IDS_PATH)
        else:
            self.dense_retriever.build_index(Config.CORPUS_PATH, Config.DENSE_INDEX_PATH, Config.DENSE_DOC_IDS_PATH)

        # 3. Initialize Reranker
        self.reranker.load_model(Config.RERANK_MODEL_NAME)

        # 4. Initialize LLM (if enabled)
        if self.generator is not None:
            self.generator.load_model(Config.LLM_MODEL_NAME)

    def retrieve_only(self, query: str) -> list:
        """
        Executes the search pipeline (BM25 + Dense -> RRF -> Rerank)
        and returns the top candidates.
        """
        # 1. Retrieve candidates
        bm25_res = self.bm25_retriever.retrieve(query, top_k=Config.TOP_K_RETRIEVE)
        dense_res = self.dense_retriever.retrieve(query, top_k=Config.TOP_K_RETRIEVE)

        # 2. Run Reciprocal Rank Fusion
        fused_res = reciprocal_rank_fusion([bm25_res, dense_res])

        # Keep top candidates for rerank
        fused_candidates = fused_res[:Config.TOP_K_RRF]

        # 3. Rerank candidates
        reranked_res = self.reranker.rerank(
            query=query,
            doc_list=fused_candidates,
            corpus=self.corpus,
            top_k=Config.TOP_K_RERANK
        )
        
        return reranked_res

    def query(self, query: str) -> dict:
        """
        Executes the full RAG pipeline (Retrieval + Generation).
        """
        # Step 1: Retrieve context
        retrieved_docs = self.retrieve_only(query)

        # Step 2: Generate Answer using LLM
        if self.generator is not None and retrieved_docs:
            answer = self.generator.generate(query, retrieved_docs, self.corpus)
        else:
            answer = "LLM generator is disabled or no documents were retrieved."

        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": [
                {
                    "corpus-id": doc["corpus-id"],
                    "score": doc["score"],
                    "text": self.corpus.get(doc["corpus-id"], "")
                }
                for doc in retrieved_docs
            ]
        }
