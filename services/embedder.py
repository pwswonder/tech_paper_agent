# services/embedder.py
from __future__ import annotations

import os
import re
import json
import math
from typing import TypedDict, List, Dict, Any, Tuple, Callable

from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable

# Embedding / VS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Hybrid Retrieval (Sparse)
from langchain_community.retrievers import BM25Retriever

# Ensemble + Compression (표준 클래스 사용)

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from typing import Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)


# Cache helpers
from services.retriever_cache import (
    has_retriever,
    get_retriever,
    get_vectorstore,
    set_retriever,
)


class EmbedState(TypedDict, total=False):
    # 입력
    raw_text: str
    meta: Dict[str, Any]
    # 출력
    raw_texts: List[str]
    chunks: List[str]
    vectorstore: FAISS
    retriever: Any
    # 옵션
    top_k: int
    document_id: int
    doc_id: int
    doc_meta: Dict[str, Any]


# ------------------------------
# Helpers / Env
# ------------------------------
def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _dbg(msg: str):
    if os.getenv("RAG_DEBUG", "0") != "0":
        print(f"[RAG] {msg}")


# helpers 섹션 어딘가에 추가
def _doc_key(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    return f"{md.get('source','')}|{md.get('chunk_id','')}|{md.get('section','')}"


def _safe_keep_dense(dense_docs, fused_docs, top_k: int, keep_m: int):
    out, seen = [], set()
    # 1) Dense 상위 m개 선점
    for d in dense_docs[: max(0, keep_m)]:
        k = _doc_key(d)
        if k and k not in seen:
            out.append(d)
            seen.add(k)
            if len(out) >= top_k:
                return out[:top_k]
    # 2) RRF 결과로 채우기
    for d in fused_docs:
        k = _doc_key(d)
        if k and k not in seen:
            out.append(d)
            seen.add(k)
            if len(out) >= top_k:
                break
    return out[:top_k]


# ------------------------------
# String utils / guards
# ------------------------------
_NOISE_RE = re.compile(
    r"(arxiv|doi:|https?://|submitted|preprint|license|creativecommons|all rights reserved|"
    r"permission|grants? permission|hereby|redistribution|use with attribution|"
    r"google|scholarly works|cc-by|cc by|copyright)",
    re.I,
)

_AFFIL_RE = re.compile(
    r"(univ|university|department|lab|laboratory|institute|school of|research)", re.I
)


def _basename_no_ext(path: str) -> str:
    if not path:
        return ""
    name = os.path.basename(path)
    return re.sub(r"\.(pdf|txt|md|docx?)$", "", name, flags=re.I)


def _looks_like_filename(s: str | None, source_path: str | None = None) -> bool:
    if not s:
        return True
    st = s.strip()
    if len(st) < 6:
        return True
    if " " not in st and len(st) <= 20:
        return True
    if (st.count("_") + st.count("-")) >= max(2, len(st) // 4):
        return True
    letters = sum(ch.isalpha() for ch in st)
    if letters / max(1, len(st)) < 0.45:
        return True
    if source_path:
        stem = _basename_no_ext(source_path)
        stem_norm = re.sub(r"[_\-]+", "", stem).lower()
        st_norm = re.sub(r"[_\-]+", "", st).lower()
        if stem_norm and stem_norm == st_norm:
            return True
    if re.search(r"\barxiv[-_ ]?\d{4}\.\d{4,5}\b", st, re.I):
        return True
    if re.search(r"\bv\d+\b", st, re.I):
        return True
    return False


def _split_first_page(raw_text: str) -> List[str]:
    first = raw_text.split("\f")[0] if "\f" in raw_text else raw_text[:10000]
    blocks, cur = [], []
    for line in first.splitlines():
        line = line.strip()
        if not line:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks[:20]


def _strip_disclaimer(blocks: List[str]) -> List[str]:
    cleaned = []
    for b in blocks:
        if (
            _NOISE_RE.search(b)
            and sum(1 for l in b.splitlines() if _NOISE_RE.search(l)) >= 1
        ):
            continue
        cleaned.append(b)
    return cleaned or blocks


def _pick_title_from_blocks(blocks: List[str]) -> str | None:
    cutoff_idx = len(blocks)
    for i, b in enumerate(blocks[:8]):
        if re.search(
            r"^\s*(abstract|요약|introduction)\b", b.splitlines()[0] if b else "", re.I
        ):
            cutoff_idx = i
            break
    cand_blocks = blocks[:cutoff_idx]

    def line_ok(l: str) -> bool:
        if len(l) < 6 or len(l) > 180:
            return False
        if "@" in l:
            return False
        if _NOISE_RE.search(l):
            return False
        if _AFFIL_RE.search(l):
            return False
        if l.count(",") >= 3:
            return False
        if re.search(r"https?://|^\[|\]$|\(|\)$", l):
            return False
        digits = sum(ch.isdigit() for ch in l)
        if digits / max(1, len(l)) > 0.2:
            return False
        return True

    candidates: List[Tuple[str, float]] = []
    for b in cand_blocks[:6]:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        for i, l in enumerate(lines[:3]):
            if not line_ok(l):
                continue
            L = len(l)
            punc = sum(1 for ch in l if ch in ".,:;!?")
            under = l.count("_") + l.count("-")
            words = len(l.split())
            cap_words = sum(1 for w in l.split() if re.match(r"^[A-Z][a-z\-]+$", w))
            score = 0.0
            score += (min(L, 120) / 120) * 2.0
            score += (cap_words / max(1, words)) * 2.0
            score -= (punc / max(1, L)) * 1.5
            score -= (under / max(1, L)) * 2.0
            candidates.append((l, score))
        if len(lines) >= 2 and line_ok(lines[0]) and line_ok(lines[1]):
            combo = f"{lines[0]}: {lines[1]}"
            if len(combo) <= 180:
                L = len(combo)
                words = len(combo.split())
                cap_words = sum(
                    1 for w in combo.split() if re.match(r"^[A-Z][a-z\-]+$", w)
                )
                score = (min(L, 120) / 120) * 2.0 + (cap_words / max(1, words)) * 2.0
                candidates.append((combo, score * 0.95))
    if not candidates:
        return None
    best = max(candidates, key=lambda x: x[1])[0]
    if _looks_like_filename(best, None):
        return None
    return re.sub(r"\s+", " ", best).strip()


def _looks_like_person_name(token: str) -> bool:
    return bool(re.match(r"^[A-Z][a-z]+(?:[- ][A-Z][a-z]+)?$", token))


def _pick_authors_from_blocks(blocks: List[str]) -> str | None:
    lines = []
    for b in blocks[:8]:
        lines.extend([l.strip() for l in b.splitlines() if l.strip()])

    def is_bad(l: str) -> bool:
        if "@" in l:
            return True
        if _NOISE_RE.search(l):
            return True
        if _AFFIL_RE.search(l):
            return True
        if len(l) < 6 or len(l) > 200:
            return True
        if any(sym in l for sym in ["http://", "https://"]):
            return True
        if sum(ch.isdigit() for ch in l) > 0:
            return True
        return False

    best_line, best_score = None, -1.0
    for l in lines[:40]:
        if is_bad(l):
            continue
        parts = [p.strip() for p in re.split(r",| and ", l) if p.strip()]
        if len(parts) < 2:
            continue
        name_like = sum(1 for p in parts if _looks_like_person_name(p))
        if name_like < 2:
            continue
        score = name_like / len(parts)
        if score > best_score:
            best_line, best_score = l, score

    if not best_line:
        return None
    parts = [p.strip() for p in re.split(r",| and ", best_line) if p.strip()]
    cleaned = [p for p in parts if _looks_like_person_name(p)]
    uniq: List[str] = []
    seen = set()
    for p in cleaned:
        if p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    if len(uniq) >= 2:
        return ", ".join(uniq)
    return None


def _extract_year_near_context(blocks: List[str]) -> str | None:
    ctx_re = re.compile(
        r"(published|conference|proceedings|arxiv|nips|neurips|iclr|icml|cvpr|acl|kdd|aaai|ijcai)",
        re.I,
    )
    years: List[int] = []
    for b in blocks[:8]:
        for l in b.splitlines():
            if ctx_re.search(l) and not _NOISE_RE.search(l):
                for m in re.finditer(r"\b(19|20)\d{2}\b", l):
                    years.append(int(m.group(0)))
    if years:
        years.sort()
        return str(years[len(years) // 2])
    return None


def _extract_ids(head: str) -> Dict[str, str]:
    out = {}
    m = re.search(r"\b10\.\d{4,9}/\S+\b", head)
    if m:
        out["doi"] = m.group(0).rstrip(" .,)];")
    m2 = re.search(r"\barXiv[: ]?(\d{4}\.\d{4,5})(v\d+)?\b", head, re.I)
    if m2:
        out["arxiv"] = m2.group(1)
    return out


def _maybe_lookup_metadata(ids: Dict[str, str]) -> Dict[str, Any]:
    if os.getenv("RAG_META_LOOKUP", "0") != "1":
        return {}
    timeout = _get_env_int("RAG_META_LOOKUP_TIMEOUT", 3)
    result: Dict[str, Any] = {}
    try:
        import requests
    except Exception:
        return {}
    try:
        if "doi" in ids:
            r = requests.get(
                f"https://api.crossref.org/works/{ids['doi']}", timeout=timeout
            )
            if r.ok:
                j = r.json().get("message", {})
                title = (j.get("title") or [None])[0]
                authors = j.get("author") or []
                author_names = []
                for a in authors:
                    n = " ".join([x for x in [a.get("given"), a.get("family")] if x])
                    if n:
                        author_names.append(n)
                year = None
                if j.get("issued", {}).get("date-parts"):
                    year = j["issued"]["date-parts"][0][0]
                venue = j.get("container-title", [None])[0]
                if title:
                    result["title"] = title
                if author_names:
                    result["authors"] = ", ".join(author_names)
                if year:
                    result["year"] = str(year)
                if venue:
                    result["venue"] = venue
                result["doi"] = ids["doi"]
                return result
    except Exception:
        pass
    try:
        if "arxiv" in ids:
            r = requests.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"id:{ids['arxiv']}",
                    "start": 0,
                    "max_results": 1,
                },
                timeout=timeout,
            )
            if r.ok:
                txt = r.text
                titles = re.findall(r"<title>([^<]+)</title>", txt)
                title = titles[1].strip() if len(titles) > 1 else None
                authors = re.findall(r"<name>([^<]+)</name>", txt)
                year = None
                y = re.search(r"<published>(\d{4})-", txt)
                if y:
                    year = y.group(1)
                if title:
                    result["title"] = title
                if authors:
                    result["authors"] = ", ".join(authors)
                if year:
                    result["year"] = str(year)
                result["venue"] = "arXiv"
                return result
    except Exception:
        pass
    return {}


# 파일 상단 어딘가에 유틸 추가
def _doc_key(doc):
    md = getattr(doc, "metadata", {}) or {}
    return f"{md.get('source','')}|{md.get('chunk_id','')}|{md.get('section','')}"


def keep_dense_topk_then_fill(dense_docs, bm25_docs, fused_docs, top_k=5, keep_m=2):
    out = []
    seen = set()
    # 1) Dense 상위 m개 선점
    for d in dense_docs[:keep_m]:
        k = _doc_key(d)
        if k and k not in seen:
            out.append(d)
            seen.add(k)
            if len(out) >= top_k:
                return out[:top_k]
    # 2) RRF 결과로 채우기
    for d in fused_docs:
        k = _doc_key(d)
        if k and k not in seen:
            out.append(d)
            seen.add(k)
            if len(out) >= top_k:
                break
    return out[:top_k]


# ------------------------------
# BM25 preprocess (NEW)
# ------------------------------
def _bm25_preprocess(t: str) -> List[str]:
    """
    - snake_case -> 공백
    - camelCase -> 공백 분절
    - 숫자/단위(224x224, 3x3, 1e-9, --batch-size) 보존
    - 기호는 최소한만 제거
    """
    t = t or ""
    s = t.lower()
    s = s.replace("_", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)  # camelCase -> 공백
    # 허용 문자 외 공백 치환 (영문/숫자/공백/.-+x/e/=/~/:/% 포함)
    s = re.sub(r"[^0-9a-z\-\+x\.\s/e=~:%]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


# ------------------------------
# Chunker
# ------------------------------
@traceable
def _build_chunks(
    raw_text: str, meta: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    meta = meta or {}
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("##", "subsection")],
        strip_headers=False,
    )
    try:
        header_docs = header_splitter.split_text(raw_text)
        base_sections = [
            {
                "text": d.page_content,
                "metadata": {"section": d.metadata.get("header", "")},
            }
            for d in header_docs
        ]
    except Exception:
        base_sections = [{"text": raw_text, "metadata": {"section": "whole_document"}}]

    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_get_env_int("RAG_CHUNK_SIZE", 1000),
        chunk_overlap=_get_env_int("RAG_CHUNK_OVERLAP", 120),
        separators=["\n\n", "\n", " ", ""],
    )

    results: List[Dict[str, Any]] = []
    for sec_id, sec in enumerate(base_sections):
        pieces = body_splitter.split_text(sec["text"])
        for i, piece in enumerate(pieces):
            md = {
                "source": meta.get("source") or meta.get("title") or "N/A",
                "title": "N/A",
                "section": sec.get("metadata", {}).get("section")
                or sec.get("section")
                or "N/A",
                "chunk_id": f"{sec_id}-{i}",
            }
            results.append({"text": piece, "metadata": md})
    return results


# ------------------------------
# Metadata extraction (robust)
# ------------------------------
def _extract_paper_metadata(raw_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "title": None,
        "authors": None,
        "year": None,
        "venue": None,
        "doi": None,
        "source": meta.get("source"),
    }
    trust_input = os.getenv("RAG_TRUST_INPUT_TITLE", "0") == "1"
    meta_title = str(meta.get("title") or "").strip()
    if (
        trust_input
        and meta_title
        and not _looks_like_filename(meta_title, meta.get("source"))
    ):
        out["title"] = re.sub(r"\s+", " ", meta_title)

    blocks = _split_first_page(raw_text)
    blocks = _strip_disclaimer(blocks)

    head = "\n".join(blocks)[:8000]
    ids = _extract_ids(head)
    if ids.get("doi"):
        out["doi"] = ids["doi"]

    if not out["title"]:
        t = _pick_title_from_blocks(blocks)
        if t and not _looks_like_filename(t, meta.get("source")):
            out["title"] = t

    a = _pick_authors_from_blocks(blocks)
    if a:
        out["authors"] = a

    y = _extract_year_near_context(blocks)
    if y:
        out["year"] = y

    for b in blocks[:8]:
        for l in b.splitlines():
            if re.search(
                r"(NIPS|NeurIPS|ICLR|ICML|CVPR|ACL|KDD|AAAI|IJCAI|Nature|Science|IEEE|ACM)",
                l,
                re.I,
            ):
                if not _NOISE_RE.search(l):
                    out["venue"] = l.strip()
                    break
        if out["venue"]:
            break

    remote = _maybe_lookup_metadata(ids)
    for k in ("title", "authors", "year", "venue", "doi"):
        if remote.get(k):
            out[k] = remote[k]

    return out


class TopKEnsembleRetriever(BaseRetriever):
    """
    EnsembleRetriever 결과(fused)를 사용하되,
    Dense 상위 keep_m개는 무조건 보존한 뒤 나머지를 RRF로 채운다.
    """

    ensemble: Any
    dense_cand: Any
    top_k: int = 5
    keep_m: int = 2

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        fused = self.ensemble.get_relevant_documents(query)
        dense_docs = self.dense_cand.get_relevant_documents(query)
        return _safe_keep_dense(dense_docs, fused, self.top_k, self.keep_m)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # 동기와 동일 동작
        return self._get_relevant_documents(query)


# ------------------------------
# Main embedder
# ------------------------------
@traceable
def embedder(state: EmbedState) -> EmbedState:
    """
    - 제목/저자 등 메타 보강
    - Dense(FAISS) + BM25 하이브리드(EnsembleRetriever/RRF)
      * 각 리트리버는 fetch_k만큼 넓게 후보 생성
      * 앙상블에서 결합 후 최종 top_k로 슬라이스
    - (옵션) EmbeddingsFilter로 경량 압축
    - 카탈로그 청크 추가
    """
    # Fast-path cache
    try:
        bypass_cache = os.getenv("RAG_BYPASS_CACHE", "0") == "1"
        doc_id = state.get("document_id") or state.get("doc_id")
        if (not bypass_cache) and (doc_id is not None) and has_retriever(int(doc_id)):
            ret = get_retriever(int(doc_id))
            vs = get_vectorstore(int(doc_id))
            top_k_env = _get_env_int("QA_TOPK", 5)
            _dbg(f"CACHE HIT: doc_id={doc_id}")
            return {
                **state,
                "retriever": ret,
                "vectorstore": vs,
                "top_k": state.get("top_k", top_k_env),
                "raw_texts": state.get("raw_texts", []),
                "chunks": state.get("chunks", []),
            }
    except Exception as e:
        _dbg(f"cache path error: {e}")

    # Input
    raw_text = (state.get("raw_text") or "").strip()
    if not raw_text:
        _dbg("raw_text empty -> abort")
        return {
            **state,
            "retriever": None,
            "vectorstore": None,
            "chunks": [],
            "raw_texts": [],
        }
    meta: Dict[str, Any] = state.get("meta", {}) or {}

    # Metadata (robust)
    doc_meta = _extract_paper_metadata(raw_text, meta)
    _dbg(
        f"doc_meta: title='{doc_meta.get('title')}' authors='{doc_meta.get('authors')}' year='{doc_meta.get('year')}' doi='{doc_meta.get('doi')}'"
    )

    # Chunks
    chunk_dicts = _build_chunks(raw_text, meta=meta)
    for c in chunk_dicts:
        if doc_meta.get("title") and not _looks_like_filename(
            doc_meta.get("title"), meta.get("source")
        ):
            c["metadata"]["title"] = doc_meta["title"]
        c["metadata"].update({k: v for k, v in doc_meta.items() if v})

    raw_texts: List[str] = [c["text"] for c in chunk_dicts]
    texts = raw_texts[:]
    metadatas = [c["metadata"] for c in chunk_dicts]

    # Catalog chunk
    if os.getenv("RAG_ADD_CATALOG", "1") != "0":
        safe_title = doc_meta.get("title")
        if _looks_like_filename(safe_title, meta.get("source")):
            safe_title = None
        cat_lines = [
            f"제목(Title): {safe_title or 'N/A'}",
            f"저자(Authors): {doc_meta.get('authors') or 'N/A'}",
            f"발표연도(Year): {doc_meta.get('year') or 'N/A'}",
            f"학회/저널(Venue): {doc_meta.get('venue') or 'N/A'}",
            f"DOI: {doc_meta.get('doi') or 'N/A'}",
            f"출처(Source): {doc_meta.get('source') or meta.get('source') or 'N/A'}",
        ]
        texts.append("\n".join(cat_lines))
        metadatas.append(
            {
                "source": doc_meta.get("source")
                or meta.get("source")
                or meta.get("title")
                or "N/A",
                "title": safe_title or "N/A",
                "section": "catalog",
                "chunk_id": "catalog-0",
                "is_catalog": True,
                **{k: v for k, v in doc_meta.items() if v},
            }
        )

    # Embeddings
    dep = os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
    endpoint = os.getenv("AOAI_ENDPOINT")
    if not dep or not endpoint:
        raise RuntimeError(
            "[RAG] AOAI_DEPLOY_EMBED_3_LARGE 또는 AOAI_ENDPOINT 가 비어 있음"
        )
    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment=dep,
        openai_api_version=os.getenv("AOAI_API_VERSION", "2024-02-01"),
        api_key=os.getenv("AOAI_API_KEY"),
        azure_endpoint=endpoint,
    )

    vectorstore = FAISS.from_texts(
        texts=texts, embedding=embedding_model, metadatas=metadatas
    )
    try:
        ntotal = vectorstore.index.ntotal
    except Exception:
        ds = getattr(vectorstore, "docstore", None)
        ntotal = len(getattr(ds, "_dict", {})) if ds and hasattr(ds, "_dict") else -1
    _dbg(f"faiss.ntotal={ntotal}")

    # Retriever params
    top_k = state.get("top_k", _get_env_int("QA_TOPK", 5))
    fetch_k = _get_env_int("RAG_FETCHK", max(20, top_k * 6))
    mmr_lambda = _get_env_float("RAG_MMR_LAMBDA", 0.5)
    rrf_c = _get_env_int("RAG_RRF_C", 60)  # Ensemble 내부의 RRF는 c≈60 기본
    weights_env = os.getenv("RAG_HYBRID_WEIGHTS", "0.6,0.4")
    try:
        w_dense, w_sparse = [float(x) for x in weights_env.split(",")]
    except Exception:
        w_dense, w_sparse = 0.6, 0.4

    # ---- 후보군 확장 ----
    # Dense 후보: similarity 로 fetch_k 확장
    dense_cand = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": fetch_k}
    )

    # BM25 후보: fetch_k 확장 + 커스텀 전처리
    try:
        bm25 = BM25Retriever.from_texts(
            [t if isinstance(t, str) else "" for t in texts],
            metadatas=metadatas,
            preprocess_func=_bm25_preprocess,  # langchain>=0.1.x
        )
    except TypeError:
        bm25 = BM25Retriever.from_texts(
            [t if isinstance(t, str) else "" for t in texts],
            metadatas=metadatas,
            preprocess_fn=_bm25_preprocess,  # fallback for older versions
        )
    bm25.k = fetch_k

    # ---- RRF 앙상블(표준 EnsembleRetriever) → 최종 top_k 슬라이스 ----
    use_hybrid = os.getenv("RAG_HYBRID", "1") != "0"
    if use_hybrid:
        ensemble = EnsembleRetriever(
            retrievers=[dense_cand, bm25],
            weights=[w_dense, w_sparse],
        )
        keep_m = _get_env_int("RAG_KEEP_DENSE_M", 2)  # ← 환경변수로 제어(기본 2)
        base_ret = TopKEnsembleRetriever(
            ensemble=ensemble,
            dense_cand=dense_cand,  # ← 추가
            top_k=top_k,
            keep_m=keep_m,  # ← 추가
        )

    else:
        # 하이브리드 끄면 Dense MMR(최종 다양화) 사용
        base_ret = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": max(fetch_k, top_k * 5),
                "lambda_mult": mmr_lambda,
            },
        )

    # ---- (유지) 경량 압축 ----
    use_compress = os.getenv("RAG_COMPRESS", "1") != "0"
    if use_compress:
        sim_th = _get_env_float("RAG_SIM_THRESHOLD", 0.76)
        compressor = EmbeddingsFilter(
            embeddings=embedding_model, similarity_threshold=sim_th
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_ret
        )
    else:
        retriever = base_ret

    _dbg(
        f"params: hybrid={use_hybrid} top_k={top_k} fetch_k={fetch_k} "
        f"keep_m={_get_env_int('RAG_KEEP_DENSE_M', 2)} weights={w_dense},{w_sparse} "
        f"compress={use_compress}"
    )

    # Probe
    try:
        probe_terms = (doc_meta.get("title") or "").split()[:6]
        probe_q = " ".join(probe_terms) if probe_terms else "저자 제목 연도 학회 DOI"
        docs = retriever.get_relevant_documents(probe_q)[:3]
        _dbg(f"probe_q='{probe_q}' -> hits={len(docs)}")
        for i, d in enumerate(docs):
            _dbg(
                f" [{i}] src={d.metadata.get('source')}, sec={d.metadata.get('section')}, len={len(d.page_content)}"
            )
    except Exception as e:
        _dbg(f"probe retrieval error: {e}")

    # Cache save
    try:
        _doc_id = state.get("document_id") or state.get("doc_id")
        if _doc_id is not None:
            set_retriever(int(_doc_id), retriever, vectorstore)
    except Exception as e:
        _dbg(f"cache save error: {e}")

    return {
        **state,
        "raw_texts": raw_texts,
        "chunks": raw_texts,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "top_k": top_k,
        "doc_meta": doc_meta,
    }


# ------------------------------
# Utility
# ------------------------------
def _state_get_doc_id(state):
    for k in ("document_id", "doc_id", "selected_doc_id", "documentId", "docId"):
        v = state.get(k)
        if v not in (None, "", 0):
            try:
                return int(str(v))
            except Exception:
                try:
                    return int(
                        "".join(ch for ch in str(v) if str(v) and str(ch).isdigit())
                    )
                except Exception:
                    pass
    meta = state.get("meta") or {}
    if isinstance(meta, dict):
        for k in ("document_id", "doc_id", "id", "selected_doc_id"):
            v = meta.get(k)
            if v not in (None, "", 0):
                try:
                    return int(str(v))
                except Exception:
                    try:
                        return int(
                            "".join(ch for ch in str(v) if str(v) and str(ch).isdigit())
                        )
                    except Exception:
                        pass
        doc = meta.get("document") or meta.get("doc")
        if isinstance(doc, dict):
            v = doc.get("id") or doc.get("document_id") or doc.get("doc_id")
            if v not in (None, "", 0):
                try:
                    return int(str(v))
                except Exception:
                    try:
                        return int(
                            "".join(ch for ch in str(v) if str(v) and str(ch).isdigit())
                        )
                    except Exception:
                        pass
    return None
