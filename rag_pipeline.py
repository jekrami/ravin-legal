import logging
import pickle

import faiss
import numpy as np
from langchain_ollama.embeddings import OllamaEmbeddings

from config import EMBEDDING_MODEL, KEYWORD_BOOST_CONFIG, TOP_K

logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.index = None
        self.documents = []
        self.document_sources = []

    def _rebuild_index(self):
        """Rebuild the FAISS index from current documents."""
        if not self.documents:
            self.index = None
            return

        embeddings = self.embedding_model.embed_documents(self.documents)
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype="float32"))

    def add_documents(self, docs, source_name="Unknown"):
        """Add documents with source tracking. Only non-empty chunks are stored."""
        valid_docs = [d for d in docs if d.strip()]
        if not valid_docs:
            logger.warning("No valid documents to add for source %s", source_name)
            return 0

        self.documents.extend(valid_docs)
        self.document_sources.extend([source_name] * len(valid_docs))

        embeddings = self.embedding_model.embed_documents(valid_docs)

        if self.index is None:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings, dtype="float32"))
        logger.info("Added %d chunks from %s", len(valid_docs), source_name)
        return len(valid_docs)

    def retrieve(self, query, return_sources=False):
        """Retrieve relevant documents with optional source tracking."""
        if not self.index or not self.documents:
            return "" if not return_sources else ("", [])

        query_embedding = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype="float32"), TOP_K
        )

        retrieved_docs_with_scores = []
        for i, doc_index in enumerate(indices[0]):
            if doc_index < len(self.documents):
                retrieved_docs_with_scores.append({
                    "doc": self.documents[doc_index],
                    "source": (
                        self.document_sources[doc_index]
                        if doc_index < len(self.document_sources)
                        else "Unknown"
                    ),
                    "semantic_score": 1 / (1 + distances[0][i]),
                })

        for item in retrieved_docs_with_scores:
            keyword_score = sum(
                boost for keyword, boost in KEYWORD_BOOST_CONFIG.items()
                if keyword in item["doc"]
            )
            item["final_score"] = item["semantic_score"] + keyword_score

        re_ranked_docs = sorted(
            retrieved_docs_with_scores, key=lambda x: x["final_score"], reverse=True
        )

        final_docs = [item["doc"] for item in re_ranked_docs]
        sources = [item["source"] for item in re_ranked_docs]

        if return_sources:
            return "\n---\n".join(final_docs), sources
        return "\n---\n".join(final_docs)

    def delete_document(self, source_name):
        """Delete all chunks from a specific source document."""
        if not self.documents:
            return False

        indices_to_keep = [
            i for i, src in enumerate(self.document_sources) if src != source_name
        ]

        if len(indices_to_keep) == len(self.documents):
            return False

        self.documents = [self.documents[i] for i in indices_to_keep]
        self.document_sources = [self.document_sources[i] for i in indices_to_keep]
        self._rebuild_index()
        logger.info("Deleted document %s; %d chunks remain", source_name, len(self.documents))
        return True

    def get_loaded_documents(self):
        """Get list of unique source documents."""
        return list(set(self.document_sources))

    def get_document_stats(self):
        """Get statistics about loaded documents."""
        stats = {}
        for source in self.document_sources:
            stats[source] = stats.get(source, 0) + 1
        return stats

    def save(self, path):
        if self.index is None:
            return
        with open(path, "wb") as f:
            pickle.dump({
                "index": faiss.serialize_index(self.index),
                "documents": self.documents,
                "document_sources": self.document_sources,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data["index"])
            self.documents = data["documents"]
            self.document_sources = data.get(
                "document_sources", ["Unknown"] * len(self.documents)
            )

        # Reconcile legacy stores where index row count may not match documents.
        if self.index and self.index.ntotal != len(self.documents):
            logger.warning(
                "FAISS index size (%d) mismatches documents (%d); rebuilding index",
                self.index.ntotal,
                len(self.documents),
            )
            self._rebuild_index()
