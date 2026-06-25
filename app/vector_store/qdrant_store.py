"""
app/vector_store/qdrant_store.py
─────────────────────────────────────────────────────────────────────────────
Qdrant vector store — concrete implementation of BaseVectorStore.

Features:
  - Collection auto-created on first upsert (no manual setup required)
  - Dense ANN search (cosine similarity)
  - Full-text keyword search via Qdrant payload index
  - Hybrid search using Reciprocal Rank Fusion (RRF)
  - Metadata filters on department / doc_type / doc_category / tags
  - Batch upsert with configurable chunk size
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.base.vector_store import BaseVectorStore
from app.core.models.document import ChunkMetadata, EmbeddedChunk, SearchResult

logger = logging.getLogger(__name__)
settings = get_settings()

_BATCH_SIZE = 100   # points per upsert call


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant-backed vector store.

    Args:
        host:            Qdrant host (default from config).
        port:            Qdrant HTTP port.
        collection_name: Collection to use (auto-created if missing).
        vector_size:     Embedding dimension (must match embedder).
        api_key:         Qdrant API key (leave empty for local dev).
    """

    def __init__(
        self,
        host:            Optional[str] = None,
        port:            Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size:     Optional[int] = None,
        api_key:         Optional[str] = None,
    ):
        self._host       = host            or settings.qdrant_host
        self._port       = port            or settings.qdrant_port
        self._collection = collection_name or settings.qdrant_collection_name
        self._vector_size = vector_size    or settings.embedding_dimension
        self._api_key    = api_key         or settings.qdrant_api_key or None
        self._client     = None            # lazy init

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert(self, chunks: List[EmbeddedChunk]) -> int:
        from qdrant_client.models import PointStruct
        client = await self._get_client()
        await self._ensure_collection()

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id)),
                vector=chunk.embedding,
                payload=self._chunk_to_payload(chunk),
            )
            for chunk in chunks
        ]

        upserted = 0
        for i in range(0, len(points), _BATCH_SIZE):
            batch = points[i : i + _BATCH_SIZE]
            await client.upsert(collection_name=self._collection, points=batch)
            upserted += len(batch)
            logger.debug("Upserted %d/%d points", upserted, len(points))

        return upserted

    async def delete(self, chunk_ids: List[str]) -> int:
        from qdrant_client.models import PointIdsList
        client = await self._get_client()
        qdrant_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)) for cid in chunk_ids]
        await client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=qdrant_ids),
        )
        return len(chunk_ids)

    async def delete_document(self, doc_id: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = await self._get_client()
        result = await client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        count = getattr(result, "operation_id", 0)
        logger.info("Deleted chunks for doc_id=%s", doc_id)
        return count

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: List[float],
        top_k:           int = 20,
        filters:         Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        client = await self._get_client()
        qdrant_filter = self._build_filter(filters)

        hits = await client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [self._hit_to_result(h, "dense") for h in hits]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Full-text search on the `text` payload field using Qdrant's
        built-in text index (BM25-like scoring).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchText
        client = await self._get_client()

        text_condition = FieldCondition(key="text", match=MatchText(text=query))
        base_filter    = self._build_filter(filters)

        if base_filter and base_filter.must:
            base_filter.must.append(text_condition)
        else:
            from qdrant_client.models import Filter as QFilter
            base_filter = QFilter(must=[text_condition])

        # Qdrant scroll with filter for text match
        results, _ = await client.scroll(
            collection_name=self._collection,
            scroll_filter=base_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_result(p, "keyword") for p in results]

    async def hybrid_search(
        self,
        query:           str,
        query_embedding: List[float],
        top_k:           int = 20,
        dense_weight:    float = 0.7,
        keyword_weight:  float = 0.3,
        filters:         Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion of dense + keyword results.
        RRF score = sum(1 / (k + rank)) for each result list.
        """
        fetch_k = top_k * 3   # fetch more before fusion
        dense_results   = await self.search(query_embedding, fetch_k, filters)
        keyword_results = await self.keyword_search(query, fetch_k, filters)
        return self._rrf_fusion(dense_results, keyword_results, top_k, dense_weight, keyword_weight)

    # ── Admin ─────────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await client.get_collections()
            return True
        except Exception:
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        try:
            client = await self._get_client()
            info   = await client.get_collection(self._collection)
            return {
                "name":          self._collection,
                "vector_size":   self._vector_size,
                "points_count":  info.points_count,
                "status":        str(info.status),
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            kwargs: Dict[str, Any] = {"host": self._host, "port": self._port}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncQdrantClient(**kwargs)
            logger.info("Qdrant client connected to %s:%s", self._host, self._port)
        return self._client

    async def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        client     = await self._get_client()
        collections = await client.get_collections()
        names      = [c.name for c in collections.collections]
        if self._collection not in names:
            logger.info("Creating Qdrant collection '%s' (dim=%d)", self._collection, self._vector_size)
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            # Create full-text index on the `text` payload field for keyword search
            await client.create_payload_index(
                collection_name=self._collection,
                field_name="text",
                field_schema="text",
            )
            logger.info("Collection '%s' created with text index.", self._collection)

    @staticmethod
    def _chunk_to_payload(chunk: EmbeddedChunk) -> Dict[str, Any]:
        m = chunk.metadata
        return {
            "chunk_id":        chunk.id,
            "doc_id":          m.doc_id,
            "doc_filename":    m.doc_filename,
            "doc_type":        m.doc_type.value,
            "text":            chunk.text,
            "page_number":     m.page_number,
            "section_heading": m.section_heading,
            "section_id":      m.section_id,
            "chunk_index":     m.chunk_index,
            "token_count":     m.token_count,
            "char_count":      m.char_count,
            "department":      m.department,
            "doc_category":    m.doc_category,
            "tags":            m.tags,
        }

    @staticmethod
    def _payload_to_metadata(payload: Dict[str, Any]) -> ChunkMetadata:
        from app.core.models.document import DocumentType
        return ChunkMetadata(
            doc_id=payload.get("doc_id", ""),
            doc_filename=payload.get("doc_filename", ""),
            doc_type=DocumentType(payload.get("doc_type", "unknown")),
            page_number=payload.get("page_number"),
            section_heading=payload.get("section_heading"),
            section_id=payload.get("section_id"),
            chunk_index=payload.get("chunk_index", 0),
            token_count=payload.get("token_count"),
            char_count=payload.get("char_count", 0),
            department=payload.get("department"),
            doc_category=payload.get("doc_category"),
            tags=payload.get("tags", []),
        )

    def _hit_to_result(self, hit, search_type: str) -> SearchResult:
        payload = hit.payload or {}
        return SearchResult(
            chunk_id=payload.get("chunk_id", str(hit.id)),
            text=payload.get("text", ""),
            score=float(hit.score),
            metadata=self._payload_to_metadata(payload),
            search_type=search_type,
        )

    def _point_to_result(self, point, search_type: str) -> SearchResult:
        payload = point.payload or {}
        return SearchResult(
            chunk_id=payload.get("chunk_id", str(point.id)),
            text=payload.get("text", ""),
            score=1.0,      # no score in scroll results
            metadata=self._payload_to_metadata(payload),
            search_type=search_type,
        )

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]):
        if not filters:
            return None
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                for v in value:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _rrf_fusion(
        dense:   List[SearchResult],
        keyword: List[SearchResult],
        top_k:   int,
        dw:      float,
        kw:      float,
        k:       int = 60,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion with per-list weighting."""
        scores: Dict[str, float] = {}
        index:  Dict[str, SearchResult] = {}

        for rank, r in enumerate(dense, 1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + dw * (1.0 / (k + rank))
            index.setdefault(r.chunk_id, r)

        for rank, r in enumerate(keyword, 1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + kw * (1.0 / (k + rank))
            index.setdefault(r.chunk_id, r)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_id, score in ranked:
            r = index[chunk_id]
            r.score = round(score, 6)
            r.search_type = "hybrid"
            results.append(r)
        return results
