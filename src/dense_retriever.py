import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.config import Config

class DenseRetriever:
    def __init__(self):
        self.model = None
        self.index = None
        self.doc_ids = []

    def build_index(self, corpus_path: str, index_path: str, doc_ids_path: str):
        """
        Builds the FAISS index by encoding documents with SentenceTransformer.
        """
        print(f"Loading dense model: {Config.DENSE_MODEL_NAME}...")
        self.model = SentenceTransformer(Config.DENSE_MODEL_NAME)
        
        texts = []
        self.doc_ids = []
        
        print(f"Reading corpus from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.doc_ids.append(item["_id"])
                texts.append(item["text"])
                
        # BGE-M3 returns L2-normalized vectors by default with normalize_embeddings=True
        # FAISS IndexFlatIP (Inner Product) is equivalent to Cosine Similarity
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        
        print(f"Creating FAISS IndexFlatIP index with dimension {dimension}...")
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        print(f"Saving FAISS index to {index_path}...")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        print(f"Saving document IDs mapping to {doc_ids_path}...")
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False)
            
        print("Dense index built and saved successfully.")

    def load_index(self, index_path: str, doc_ids_path: str):
        """
        Loads the pre-built FAISS index and document ID mappings.
        """
        print(f"Loading dense model for querying: {Config.DENSE_MODEL_NAME}...")
        self.model = SentenceTransformer(Config.DENSE_MODEL_NAME)
        
        print(f"Loading FAISS index from {index_path}...")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading document IDs mapping from {doc_ids_path}...")
        if not os.path.exists(doc_ids_path):
            raise FileNotFoundError(f"Document IDs file not found at {doc_ids_path}")
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
            
        print("Dense index loaded successfully.")

    def retrieve(self, query: str, top_k: int = Config.TOP_K_RETRIEVE) -> list:
        """
        Retrieves top_k documents matching the query.
        Returns a list of dicts: [{'corpus-id': doc_id, 'score': score}]
        """
        if self.model is None or self.index is None:
            raise ValueError("Dense index is not loaded. Call load_index() or build_index() first.")
            
        # Encode query (make sure to normalize for Inner Product / Cosine Similarity)
        query_vector = self.model.encode(
            [query], 
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        # Search index
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: # FAISS padding if not enough candidates found
                continue
            doc_id = self.doc_ids[idx]
            results.append({
                "corpus-id": doc_id,
                "score": float(score)
            })
            
        return results
