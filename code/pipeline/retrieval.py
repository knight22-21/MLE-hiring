import re
import hashlib
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .ingestion import Document
from config import (
    logger, MAX_RETRIEVED_DOCS,
    MODEL_CACHE_DIR, EMBEDDING_CACHE_PATH, CORPUS_HASH_PATH
)


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

    def _compute_corpus_hash(self) -> str:
        """Hash all document contents to detect corpus changes."""
        h = hashlib.sha256()
        for doc in self.documents:
            h.update(doc.content.encode("utf-8"))
        return h.hexdigest()

    def _load_cached_embeddings(self, corpus_hash: str) -> bool:
        """Try to load pre-computed embeddings from disk. Returns True if successful."""
        try:
            if EMBEDDING_CACHE_PATH.exists() and CORPUS_HASH_PATH.exists():
                stored_hash = CORPUS_HASH_PATH.read_text().strip()
                if stored_hash == corpus_hash:
                    self.dense_embeddings = np.load(str(EMBEDDING_CACHE_PATH))
                    logger.info(f"Loaded cached embeddings ({self.dense_embeddings.shape[0]} vectors).")
                    return True
                else:
                    logger.info("Corpus changed since last cache. Re-computing embeddings.")
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
        return False

    def _save_embeddings_cache(self, embeddings_np: np.ndarray, corpus_hash: str):
        """Save embeddings and corpus hash to disk."""
        try:
            np.save(str(EMBEDDING_CACHE_PATH), embeddings_np)
            CORPUS_HASH_PATH.write_text(corpus_hash)
            logger.info("Saved embedding cache to disk.")
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    def _build_index(self):
        # ── BM25 ──
        logger.info("Building BM25 index...")
        tokenized_corpus = [self._tokenize(doc.content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

        # ── Dense embeddings ──
        corpus_hash = self._compute_corpus_hash()

        # Try loading from cache first
        if self._load_cached_embeddings(corpus_hash):
            # Still need the encoder for query encoding
            try:
                self.encoder = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    cache_folder=str(MODEL_CACHE_DIR)
                )
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model: {e}")
            return

        # No cache — compute from scratch
        logger.info("Computing dense embeddings with all-MiniLM-L6-v2...")
        try:
            self.encoder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=str(MODEL_CACHE_DIR)
            )
            corpus_texts = [doc.content for doc in self.documents]
            embeddings_np = self.encoder.encode(corpus_texts, show_progress_bar=True)
            self.dense_embeddings = embeddings_np
            self._save_embeddings_cache(embeddings_np, corpus_hash)
            logger.info("Dense index built and cached successfully.")
        except Exception as e:
            logger.error(f"Failed to build dense index: {e}")

    def search(self, query: str, top_k: int = MAX_RETRIEVED_DOCS) -> List[Document]:
        if not self.bm25:
            return []

        # ── BM25 Scores ──
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # ── Dense Scores ──
        if self.encoder is not None and self.dense_embeddings is not None:
            query_embedding = self.encoder.encode(query)
            # Cosine similarity via dot product (embeddings are already normalized by MiniLM)
            dense_scores = np.dot(self.dense_embeddings, query_embedding)
        else:
            dense_scores = np.zeros(len(self.documents))

        # ── Normalize ──
        bm25_max = np.max(bm25_scores)
        bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

        dense_max = np.max(dense_scores)
        dense_norm = dense_scores / dense_max if dense_max > 0 else dense_scores

        # ── Weighted combination (50/50) ──
        combined_scores = 0.5 * bm25_norm + 0.5 * dense_norm

        # ── Top-K ──
        top_indices = sorted(
            range(len(combined_scores)),
            key=lambda i: combined_scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                results.append(self.documents[idx])

        return results
