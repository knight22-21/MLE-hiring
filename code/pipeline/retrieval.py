import re
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .ingestion import Document
from config import logger, MAX_RETRIEVED_DOCS

class HybridRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.bm25 = None
        self.encoder = None
        self.dense_embeddings = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = re.split(r'\W+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def _build_index(self):
        logger.info("Building BM25 index...")
        tokenized_corpus = [self._tokenize(doc.content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")
        
        logger.info("Building Dense index with all-MiniLM-L6-v2...")
        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            corpus_texts = [doc.content for doc in self.documents]
            self.dense_embeddings = self.encoder.encode(corpus_texts, convert_to_tensor=True)
            logger.info("Dense index built successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")

    def search(self, query: str, top_k: int = MAX_RETRIEVED_DOCS) -> List[Document]:
        if not self.bm25:
            return []
            
        # BM25 Scores
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Dense Scores
        if self.encoder is not None and self.dense_embeddings is not None:
            import torch
            query_embedding = self.encoder.encode(query, convert_to_tensor=True)
            from sentence_transformers.util import cos_sim
            dense_scores = cos_sim(query_embedding, self.dense_embeddings)[0].cpu().numpy()
        else:
            dense_scores = np.zeros(len(self.documents))
            
        # Normalize scores
        if np.max(bm25_scores) > 0:
            bm25_norm = bm25_scores / np.max(bm25_scores)
        else:
            bm25_norm = bm25_scores
            
        if np.max(dense_scores) > 0:
            dense_norm = dense_scores / np.max(dense_scores)
        else:
            dense_norm = dense_scores
            
        # Simple RRF / weighted sum (50/50)
        combined_scores = 0.5 * bm25_norm + 0.5 * dense_norm
        
        # Get top k indices
        top_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0: 
                results.append(self.documents[idx])
                
        return results
