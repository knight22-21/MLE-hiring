import re
from typing import List
from rank_bm25 import BM25Okapi
from .ingestion import Document
from config import logger, MAX_RETRIEVED_DOCS

class HybridRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.bm25 = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        if not text:
            return []
        text = text.lower()
        # Basic split by non-alphanumeric
        tokens = re.split(r'\W+', text)
        return [t for t in tokens if len(t) > 1]

    def _build_index(self):
        logger.info("Building BM25 index...")
        tokenized_corpus = [self._tokenize(doc.content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")
        
        # Note: Dense embeddings (all-MiniLM-L6-v2) will be added in Phase 3.

    def search(self, query: str, top_k: int = MAX_RETRIEVED_DOCS) -> List[Document]:
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        # Get top-k scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0: # Only return docs that actually matched something
                results.append(self.documents[idx])
                
        return results
