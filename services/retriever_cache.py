# backend/services/retriever_cache.py
# 문서별 retriever/vectorstore를 메모리에 보관 (개발/PoC 용)
# 운영에서는 디스크에 FAISS 저장 또는 외부 벡터DB 권장.

from typing import Dict, Any, Optional
import logging
from typing import Any, Dict, Optional


_RETRIEVER_CACHE: Dict[int, Dict[str, Any]] = {}


def set_retriever(doc_id: int, retriever: Any, vectorstore: Any) -> None:
    _RETRIEVER_CACHE[doc_id] = {"retriever": retriever, "vectorstore": vectorstore}


def get_retriever(doc_id: int) -> Optional[Any]:
    item = _RETRIEVER_CACHE.get(doc_id)
    return item.get("retriever") if item else None


def has_retriever(doc_id: int) -> bool:
    return doc_id in _RETRIEVER_CACHE


def clear_retriever(doc_id: int) -> None:
    _RETRIEVER_CACHE.pop(doc_id, None)


def get_vectorstore(doc_id: int) -> Optional[Any]:
    it = _RETRIEVER_CACHE.get(doc_id)
    vs = it.get("vectorstore") if it else None
    logging.getLogger(__name__).debug("[rc.get_vs] doc_id=%s hit=%s", doc_id, bool(vs))
    return vs


def get_pair(doc_id: int) -> Optional[Dict[str, Any]]:
    pair = _RETRIEVER_CACHE.get(doc_id)
    logging.getLogger(__name__).debug("[rc.get_pair] doc_id=%s hit=%s", doc_id, bool(pair))
    return pair


def debug_dump() -> Dict[int, Dict[str, Any]]:
    """Return a shallow copy for debugging purpose."""
    logging.getLogger(__name__).debug("[rc.dump] size=%s", len(_RETRIEVER_CACHE))
    return dict(_RETRIEVER_CACHE)
