import json
import pickle
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from src.config import Config
from src.tokenizer import tokenize_vietnamese, tokenize_query

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []
        self.corpus = {} # mapping from doc_id to raw text

    def build_index(self, corpus_path: str, save_path: str):
        """
        Builds the BM25 index by tokenizing the corpus and fitting BM25Okapi.
        """
        print(f"Building BM25 index from {corpus_path}...")
        tokenized_corpus = []
        self.doc_ids = []
        self.corpus = {}

        # Read corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading and tokenizing corpus"):
                item = json.loads(line)
                doc_id = item["_id"]
                raw_text = item["text"]
                
                # Tokenize text
                tokenized_text = tokenize_vietnamese(raw_text)
                # Split by space to get individual tokens/words
                tokens = tokenized_text.lower().split()
                
                tokenized_corpus.append(tokens)
                self.doc_ids.append(doc_id)
                self.corpus[doc_id] = raw_text

        print("Fitting BM25Okapi model...")
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=Config.BM25_K1,
            b=Config.BM25_B
        )

        print(f"Saving BM25 index to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "doc_ids": self.doc_ids,
                "corpus": self.corpus
            }, f)
        print("BM25 index built and saved successfully.")

    def load_index(self, save_path: str):
        """
        Loads a pre-built BM25 index from disk.
        """
        print(f"Loading BM25 index from {save_path}...")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"BM25 index file not found at {save_path}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.doc_ids = data["doc_ids"]
            self.corpus = data["corpus"]
        print("BM25 index loaded successfully.")

    def retrieve(self, query: str, top_k: int = Config.TOP_K_RETRIEVE) -> list:
        """
        Retrieves top_k documents matching the query.
        Returns a list of dicts: [{'corpus-id': doc_id, 'score': score}]
        """
        if self.bm25 is None:
            raise ValueError("BM25 index is not loaded. Call load_index() or build_index() first.")

        # Tokenize query
        tok_query = tokenize_query(query).lower().split()
        
        # Calculate BM25 scores
        scores = self.bm25.get_scores(tok_query)
        
        # Zip, sort, and return top_k
        results = []
        for doc_id, score in zip(self.doc_ids, scores):
            results.append({
                "corpus-id": doc_id,
                "score": float(score)
            })
            
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
