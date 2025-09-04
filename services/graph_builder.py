# services/graph_builder.py
# ------------------------------------------------------------
# LangGraph 파이프라인 (업데이트 버전: 도메인 무관, 항상 모델/코드 생성):
# 1) embedder        : 문서 파싱/임베딩/벡터스토어 생성 및 retriever 세팅
# 2) summary_node    : 업로드 즉시 논문 요약 (리팩토링된 summarizer_agent 사용)
# 3) classify_node   : 기술 도메인 분류 (참고용. 분기에는 사용하지 않음 / 유지 선택사항)
# 4) model_extractor : 원문에서 모델 설명 섹션을 추출/분석하여 모델 정보 JSON 생성
# 5) base_code       : 모델 정보 기반으로 TensorFlow(Keras) base code 생성 (템플릿 기반)
# 6) qa_node         : 사용자 질문이 들어온 경우에만 실행 (리팩토링된 qa_agent 사용)
# ------------------------------------------------------------

from __future__ import annotations
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any, Optional

# 기존 서비스 (유지)
from services.summarizer import summarizer_agent, qa_agent
from services.classifier import classifier_agent
from services.embedder import embedder

import importlib, inspect, sys
from services import routing as _rt

# ✅ 모델 스펙 추출/코드 생성 연동
# [CHANGED] 기존: run_model_extractor / base_code_generator_agent.generate_base_code
#          변경: extract_model_spec / basecode_service.generate_base_code
from services.model_extractor_agent import extract_model_spec  # [CHANGED]
from services.basecode_service import (
    generate_base_code,
    generate_base_code_auto,
    decide_model_key_from_spec,
)

import logging
import os


logger = logging.getLogger(__name__)

# logger = logging.getLogger
from services.logging_utils import get_logger

log = get_logger("services.graph_builder")  # 파일별 고유 이름 권장

# === Helpers: slot aliasing / normalization / injection / compile harmonizer ===
import re as _re

_SLOT_ALIASES = {
    "model_head": "imports_extra",
    "imports": "imports_extra",
    "compile": "compile_override",
    "encoder": "encoder_layers",
    "encoder_blocks": "encoder_layers",
    "decoder": "decoder_layers",
    "decoder_blocks": "decoder_layers",
}

_SLOT_ANY_RE = _re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?:#\s*)?"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"
    r"(?:\{\{CUSTOM_BLOCK:\s*(?P<name1>[A-Za-z0-9_\-]+)\s*\}\}|\{CUSTOM_BLOCK:\s*(?P<name2>[A-Za-z0-9_\-]+)\s*\})"
    r"(?:\s*\{\%\s*endraw\s*\%\})?\s*$"
)


def _alias_payload_keys(payloads: dict) -> dict:
    if not isinstance(payloads, dict):
        return {}
    return {_SLOT_ALIASES.get(k, k): v for k, v in payloads.items()}


def _normalize_compile_payload_vars(payloads: dict) -> dict:
    """Normalize compile payload text: opt->optimizer, newline guards for loss_fn/metrics."""
    if not isinstance(payloads, dict):
        return {}
    txt = payloads.get("compile_override")
    if not isinstance(txt, str):
        return payloads
    t2 = _re.sub(r"(?m)^\s*opt\s*=", "optimizer = ", txt)
    t2 = _re.sub(r"(?<!\n)(loss_fn\s*=\s*)", r"\n\1", t2)
    t2 = _re.sub(r"(?<!\n)(metrics\s*=\s*)", r"\n\1", t2)
    t2 = _re.sub(r"[ \t]+\n", "\n", t2)
    out = dict(payloads)
    out["compile_override"] = t2
    return out


def _inject_slots_anystyle(text: str, payloads: dict) -> str:
    if not isinstance(payloads, dict) or not payloads:
        return text

    def _repl(m):
        indent = m.group("indent") or ""
        name = (m.group("name1") or m.group("name2") or "").strip()
        repl = payloads.get(name)
        if repl is None or repl == "":
            return m.group(0)
        return "\n".join(indent + ln for ln in str(repl).splitlines())

    return _SLOT_ANY_RE.sub(_repl, text)


def _ensure_compile_aligned(py_src: str) -> str:
    """
    Ensure model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics).
    If no compile exists, insert before 'return model'.
    """
    target = "model.compile("
    i = py_src.find(target)
    if i >= 0:
        # replace first call's args only (conservative)
        start_args = i + len(target)
        depth = 1
        j = start_args
        n = len(py_src)
        while j < n and depth > 0:
            ch = py_src[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        if depth == 0:
            before = py_src[:start_args]
            after = py_src[j - 1 :]
            repl = "optimizer=optimizer, loss=loss_fn, metrics=metrics"
            return before + repl + after
        return py_src

    # insert before return model (best-effort)
    lines = py_src.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().startswith("return model"):
            indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
            lines.insert(
                idx,
                indent
                + "model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)",
            )
            return "\n".join(lines)
    return (
        py_src + "\nmodel.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)\n"
    )


(__name__)


class AgentState(TypedDict, total=False):
    user_input: str
    raw_text: str
    raw_texts: List[str]  # 여러 청크 텍스트 (요약 품질↑)
    chunks: list
    vectorstore: any
    retriever: any
    meta: Dict[str, Any]  # 문서 메타: {"title": "...", "source": "..."} 등

    chat_history: Annotated[list, "Chat History"]
    summary: str
    domain: str
    answer: str
    top_k: int

    # ===== 모델/코드 결과 (기존 필드 유지) =====
    used_model: Optional[str]  # 예: "LSTM Autoencoder", "Vision Transformer"
    base_code: Optional[str]  # Keras base code (텍스트, 호환용)

    # ===== 내부 전달/디버깅용 (기존 주석 유지) =====
    _model_components: List[
        str
    ]  # 예: ["CNN backbone","Transformer encoder","MLP head"]
    _model_description: str  # 모델 요약 설명 문자열

    # ===== [ADDED] 템플릿 기반 codegen 연동을 위한 필드 =====
    spec: Dict[str, Any]  # [ADDED] 검증/정규화된 모델 스펙 (Phase 1 산출)
    basecode_py_path: str  # [ADDED] .generated/{model_key}_generated.py
    basecode_source: str  # [ADDED] 생성된 파이썬 소스
    basecode_summary: str  # [ADDED] Keras model.summary() 출력
    basecode_error: str  # [ADDED] 실패 시 에러 메시지


# === Robust state access helpers ===
def _state_get_doc_id(state):
    # direct keys
    for k in ("document_id", "doc_id", "selected_doc_id", "documentId", "docId"):
        v = state.get(k)
        if v not in (None, "", 0):
            try:
                return int(str(v))
            except Exception:
                try:
                    return int("".join(ch for ch in str(v) if str(ch).isdigit()))
                except Exception:
                    pass
    # nested in meta
    meta = state.get("meta") or {}
    if isinstance(meta, dict):
        for k in ("document_id", "doc_id", "id", "selected_doc_id"):
            v = meta.get(k)
            if v not in (None, "", 0):
                try:
                    return int(str(v))
                except Exception:
                    try:
                        return int("".join(ch for ch in str(v) if str(ch).isdigit()))
                    except Exception:
                        pass
        # common nesting: meta["document"] = {"id": ...}
        doc = meta.get("document") or meta.get("doc")
        if isinstance(doc, dict):
            v = doc.get("id") or doc.get("document_id") or doc.get("doc_id")
            if v not in (None, "", 0):
                try:
                    return int(str(v))
                except Exception:
                    try:
                        return int("".join(ch for ch in str(v) if str(ch).isdigit()))
                    except Exception:
                        pass
    return None


def _state_get_user_query(state):
    for k in (
        "user_input",
        "question",
        "query",
        "prompt",
        "q",
        "message",
        "text",
        "input",
    ):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # nested in meta (fallback)
    meta = state.get("meta") or {}
    if isinstance(meta, dict):
        for k in ("user_input", "question", "query"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


# === QA Fast-path (non-destructive): use retriever cache to skip heavy stages ===
def _should_qa_fast(state) -> bool:
    """
    True if (A) 질문 + (해당 문서 retriever 캐시)  OR  (B) 질문 + state에 retriever 직접 포함
    """
    import os, logging

    if str(os.getenv("QA_FASTPATH", "on")).lower() in ("0", "off", "false", "no"):
        logging.getLogger(__name__).info("[qa.fastpath] disabled via env QA_FASTPATH")
        return False

    from services.retriever_cache import has_retriever

    # 질문 텍스트 탐지
    q = None
    for k in (
        "user_input",
        "question",
        "query",
        "prompt",
        "q",
        "message",
        "text",
        "input",
    ):
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            break

    # 문서 ID 탐지 (직접/메타)
    doc_id = None
    for k in ("document_id", "doc_id", "selected_doc_id", "documentId", "docId"):
        v = state.get(k)
        if v not in (None, "", 0):
            try:
                doc_id = int(str(v))
                break
            except Exception:
                pass
    if doc_id is None:
        meta = state.get("meta") or {}
        if isinstance(meta, dict):
            v = (
                meta.get("document_id")
                or meta.get("doc_id")
                or (meta.get("document") or {}).get("id")
            )
            if v not in (None, "", 0):
                try:
                    doc_id = int(str(v))
                except Exception:
                    doc_id = None

    # (A) 캐시 히트, (B) state에 retriever 직접 포함
    has_cache = has_retriever(int(doc_id)) if doc_id is not None else False
    has_inline = bool(
        state.get("retriever")
    )  # ← 이미 임베더 결과가 state에 있으면 True
    fast = bool(q) and (has_cache or has_inline)

    logging.getLogger(__name__).info(
        "[qa.fastpath.check] user_query=%s doc_id=%s cache=%s inline=%s -> %s; keys=%s",
        bool(q),
        doc_id,
        has_cache,
        has_inline,
        fast,
        list(state.keys()),
    )
    return fast


# [ADDED] 라우팅용 키 정규화 + 키워드 보강 헬퍼
def _canonicalize_spec_for_routing(spec_in: dict, state: dict) -> dict:
    """
    - 업스트림에서 키 이름이 들쑥날쑥해도 라우팅에 필요한 4대 키를 보장:
      proposed_model_family / subtype / task_type / data_modality
    - evidence/title/summary/key_blocks로 kw_blob을 만들어 LLM 보조/라우팅에 제공
    - 논문 제목/증거 텍스트 기반의 얕은 추정(Transformer/MT/Text)을 **보수적으로** 추가
    """
    s = dict(spec_in or {})

    def pick(*names, default=""):
        for n in names:
            v = s.get(n)
            if v:
                return v
            if isinstance(state, dict):
                vs = state.get(n)
                if vs:
                    return vs
        return default

    # 1) 기본 필드 수집 (기존 spec 보존)
    family = pick("proposed_model_family", "family", "model_family")
    subtype = pick("subtype", "arch_subtype")
    task = pick("task_type", "task", "objective_task")
    modality = pick("data_modality", "modality", "input_modality")

    # 2) evidence/title/summary/key_blocks 취합 → kw_blob 후보
    texts = []
    ev = s.get("evidence") or state.get("evidence")
    if isinstance(ev, list):
        for e in ev:
            t = (e.get("text") if isinstance(e, dict) else str(e)) or ""
            if t:
                texts.append(t)
    for k in ("title", "abstract", "summary"):
        v = pick(k)
        if v:
            texts.append(str(v))
    kb = s.get("key_blocks") or state.get("key_blocks")
    if isinstance(kb, (list, tuple)):
        texts += [str(x) for x in kb]
    kw_blob = " ".join(texts)[:4000]
    low = kw_blob.lower()

    # 2.1) modality 보정: 이미지/텍스트 등 키워드로 최소 보강
    if not modality:
        if any(
            x in low
            for x in [
                "image",
                "vision",
                "pixel",
                "rgb",
                "unet",
                "u-net",
                "segmentation",
            ]
        ):
            modality = "image"
        elif any(
            x in low for x in ["token", "sentence", "bpe", "text", "nlp", "corpus"]
        ):
            modality = "text"

    # 3) 얕은 규칙 기반 보강(누락 시에만, **보수적**)
    if not family and any(
        x in low
        for x in [
            "transformer",
            "self-attention",
            "multi-head",
            "attention is all you need",
        ]
    ):
        family = "Transformer"

    # ⚠️ 번역(task_type) 승격은 텍스트 모달리티일 때만
    if not task and any(
        x in low for x in ["machine translation", "translation", "seq2seq"]
    ):
        if str(modality).lower() in {"text", "nlp", ""}:
            # 텍스트 힌트가 있거나 modality 미검출 상태일 때만 MT로 승격
            text_hint = any(x in low for x in ["token", "sentence", "bpe", "text"])
            if modality or text_hint:
                task = "machine_translation"
        # 이미지/멀티모달은 승격 금지(스키마 상 미지원 image translation은 other로 둠)

    # 4) 결과 병합
    s["proposed_model_family"] = family
    s["subtype"] = subtype
    s["task_type"] = task
    s["data_modality"] = modality
    s["_evidence_texts"] = texts
    s["kw_blob"] = kw_blob
    return s


# ------------------------------------------------------------
# 노드 함수 정의
# ------------------------------------------------------------
def model_extractor_node(state: AgentState) -> AgentState:
    """
    raw_text에서 모델 설명을 추출 → 모델 스펙(JSON) 구조화.
    [CHANGED] run_model_extractor → extract_model_spec로 변경
    결과:
      - state.used_model (호환용, 템플릿 model_key로 설정)
      - state.spec       (템플릿 codegen에 직접 사용)
      - state._model_components, state._model_description (있으면 유지)
    """
    paper_text = state.get("raw_text", "") or ""
    title = (state.get("meta") or {}).get("title")

    if not paper_text:
        log.warning("model_extractor_node: empty raw_text")
        return state

    try:
        # [CHANGED] 검증/정규화된 스펙 반환 구조에 맞춰 사용
        # [ADDED] try strict first, fallback to lenient to avoid UI breakage
        try:
            info = extract_model_spec(paper_text, title=title, strict=True)
        except Exception as _ex:
            log.warning(
                "model_extractor_node: strict validation failed (%s); retry lenient.",
                type(_ex).__name__,
            )
            try:
                info = extract_model_spec(paper_text, title=title, strict=False)
            except Exception as _ex2:
                log.error("model_extractor_node: lenient extract failed: %s", _ex2)
                info = None
        # {"raw": {...}, "verified": {...}}

        if not info:
            log.warning("model_extractor_node: no info")
            return state

        verified = info.get("verified") or {}

        # [ADDED] 템플릿 model_key 추출 휴리스틱 (스키마에 따라 다를 수 있어 방어적으로 처리)
        model_key = (
            verified.get("model_key")
            or verified.get("template")
            or verified.get("family")
            or verified.get("arch")
            or verified.get("model")
            or "cnn"
        )
        state["used_model"] = str(model_key)  # 기존 필드 유지 (표시/호환용)
        state["spec"] = verified  # [ADDED] 템플릿 codegen에 직접 사용

        # (선택) 내부 설명 필드 유지: extractor가 제공한다면 보존
        state["_model_components"] = verified.get("components", [])
        state["_model_description"] = verified.get("description", "")

    except Exception as e:
        log.exception("model_extractor_node error")
        state["basecode_error"] = f"model_extractor_node error: {e}"

    return state


def base_code_node(state: AgentState) -> AgentState:
    """
    모델 스펙(spec)과 model_key를 사용해 템플릿 기반 codegen을 수행.
    [CHANGED] LLM 생성 코드 → 템플릿+Jinja+codegen으로 전환
    결과:
      - basecode_py_path, basecode_source, basecode_summary
      - (호환) base_code: 생성 소스를 그대로 담아둠
    """
    spec = state.get("spec")
    model_key = state.get("used_model")

    # [MODIFIED] spec만 필수로 체크 (model_key는 자동 라우팅으로 보완 가능)
    if not isinstance(spec, dict):
        state["basecode_error"] = "base_code_node: missing spec"
        return state

    # [ADDED] spec 정규화 + 보강 (라우팅 직전)
    try:
        # 기존 프로젝트에 이미 존재하는 정규화 훅을 그대로 사용
        spec = _canonicalize_spec_for_routing(spec, state)
        # doc_id 주입(영속 경로 및 재수화에 필요)
        doc_id = state.get("document_id") or state.get("doc_id")
        if doc_id:
            spec["doc_id"] = str(doc_id)
        # dims 보정(템플릿에서 기대)
        spec.setdefault("dims", {})
    except Exception as e:
        log.warning("[basecode] spec canonicalize failed: %s", e)

    # ============================================
    # [ADDED] 최소 오토 라우팅 가드
    # - model_key가 없거나, Transformer 번역 스펙인데 cnn/mlp로 지정된 경우 자동 라우팅 사용
    # ============================================
    family = (
        (spec.get("proposed_model_family") or spec.get("family") or "").strip().lower()
    )
    task = (spec.get("task_type") or "").strip().lower()
    subtype = (spec.get("subtype") or "").strip().lower()

    is_transformer_mt = (family in {"transformer", "transformer_mt"}) and (
        subtype in {"encoderdecoder", "seq2seq"}
        or task in {"machine_translation", "sequence_to_sequence"}
    )
    force_auto = (model_key is None) or (
        is_transformer_mt and str(model_key).strip().lower() in {"cnn", "mlp"}
    )

    try:
        # ───────────────────────────────────────────────────────────
        # [ADDED][routing.debug] 실제 로드된 routing 모듈/함수 위치, 라우팅 직전 스냅샷
        #  - kw_blob은 절대 로컬 변수명으로 쓰지 않고 _kw_head라는 임시 변수만 사용 (스코프 충돌/UnboundLocalError 예방)
        # ───────────────────────────────────────────────────────────
        import importlib, inspect, sys
        from services import routing as _rt

        try:
            importlib.reload(_rt)  # watchfiles/streamlit 환경에서 최신 코드 로드 보장
        except Exception:
            pass

        _kw_head = ""
        try:
            _kb = spec.get("kw_blob")
            if isinstance(_kb, str) and _kb.strip():
                _kw_head = _kb[:120]
            else:
                _kw_head = str(spec.get("title") or "")[:120]
        except Exception:
            _kw_head = str(spec.get("title") or "")[:120]

        log.warning(
            "[routing.debug] module=%s resolve_line=%s",
            getattr(_rt, "__file__", "?"),
            (
                inspect.getsourcelines(_rt.resolve_template_from_spec)[1]
                if hasattr(_rt, "resolve_template_from_spec")
                else "?"
            ),
        )
        log.warning(
            "[routing.debug] pre-route snapshot: fam=%r proposed=%r task=%r mod=%r kw_head=%r",
            spec.get("family"),
            spec.get("proposed_model_family"),
            spec.get("task_type"),
            (spec.get("data_modality") or spec.get("modality")),
            _kw_head,
        )

        # ───────────────────────────────────────────────────────────
        # [EXISTING] 라우팅 호출
        # ───────────────────────────────────────────────────────────
        try:
            from services import routing

            routed_key, meta = routing.resolve_template_from_spec(spec)
            log.info("[routing] %r", (routed_key, meta))
        except Exception as e:
            routed_key, meta = (None, None)
            log.warning("[routing] failed: %s", e)

        # 라우팅 결과 채택 (+ 최소 오토 가드 적용)
        if routed_key:
            chosen_key = routed_key
        else:
            from services.basecode_service import decide_model_key_from_spec

            chosen_key = decide_model_key_from_spec(spec)

        if force_auto and routed_key:
            chosen_key = routed_key

        log.info(
            "[basecode] route resolved model_key=%s (family=%s task=%s subtype=%s)",
            chosen_key,
            family,
            task,
            subtype,
        )

        # ───────────────────────────────────────────────────────────
        # [REVISED][routing.guard] kw_blob 기반 강제 LLM 폴백
        #  - title/family 변형 여부와 무관하게 kw_blob에 이미지 번역/디퓨전 단서가 있으면 transformer를 폴백시킴
        #  - 'kw_blob' 이름을 로컬로 바인딩하지 않고 별도 임시명을 사용(UnboundLocalError 예방)
        # ───────────────────────────────────────────────────────────
        _kw_blob_local = str(spec.get("kw_blob") or "")
        _kw_low = _kw_blob_local.lower()

        # 모달리티 유추: data_modality/modality 우선, 없으면 kw_blob로 추정
        modality_eff = (
            (spec.get("data_modality") or spec.get("modality") or "").strip().lower()
        )
        if not modality_eff:
            if any(
                x in _kw_low
                for x in (
                    "image",
                    "vision",
                    "pixel",
                    "rgb",
                    "unet",
                    "u-net",
                    "segmentation",
                )
            ):
                modality_eff = "image"

        # 이미지 번역/디퓨전 힌트
        img_xfer = any(
            k in _kw_low
            for k in ("image translation", "image-to-image", "style transfer")
        )
        diff_hint = any(k in _kw_low for k in ("diffusion", "ddpm", "score-based"))

        # family/proposed가 무엇이든, kw_blob이 이미지 번역/디퓨전이면 Transformer를 강제로 폴백
        if (modality_eff in {"image", "vision", "multimodal"}) and (
            img_xfer or diff_hint
        ):
            if (
                str(chosen_key).lower() in {"transformer", "transformer_mt"}
                or not chosen_key
            ):
                log.warning(
                    "[routing.guard] kw_blob-triggered override → 'LLM 풀백' (was %r) kw_head=%r",
                    chosen_key,
                    _kw_blob_local[:80],
                )
                chosen_key = "LLM 풀백"

        # [ADDED] 실제 코드 생성 호출 (자동 라우팅 경로로 통일)
        from services.basecode_service import generate_base_code

        py_path, py_src, msum = generate_base_code(chosen_key, spec)

        # --------------------------------------------------------------
        # [ADDED] codegen 헤더에서 source_mode / template_key 파싱
        #   - write_model_file()가 '# [info] source_mode=... template_key=...' 주석을 남김
        #   - 없으면 기본값('template', chosen_key)
        # --------------------------------------------------------------
        import re as _re

        resolved_mode = "template"
        resolved_tkey = str(chosen_key)

        m = _re.search(
            r"^\s*#\s*\[info\]\s*source_mode=([^\s]+)\s+template_key=([^\s]+)",
            py_src,
            _re.M,
        )
        if m:
            try:
                resolved_mode = (m.group(1) or "template").strip().lower()
                resolved_tkey = (m.group(2) or str(chosen_key)).strip()
            except Exception:
                pass

        # 상태에 기록
        state["basecode_source_mode"] = resolved_mode
        state["template_key"] = resolved_tkey

        # === 3.4) LangGraph-style Reflection (optional) =========================
        try:
            import os
            from services.langgraph_reflection import run_langgraph_reflection

            use_lg_reflect = str(
                os.getenv("USE_LANGGRAPH_REFLECTION", "false")
            ).lower() in ("1", "true", "yes")
            if use_lg_reflect:
                lg = run_langgraph_reflection(
                    py_src=py_src,
                    spec=state.get("spec", {}),
                    max_rounds=int(os.getenv("LG_REFLECTION_ROUNDS", "2")),
                )
                # 소스 교체 및 진단 기록
                py_src = lg.get("src", py_src)
                state["langgraph_reflection"] = {
                    "slot_payloads": lg.get("slot_payloads", {}),
                    "round": lg.get("round", 0),
                    "errors": lg.get("errors", ""),
                }
        except Exception as _e:
            # 실패해도 치명적이지 않음. 원본 소스로 계속 진행.
            state["langgraph_reflection"] = {"error": str(_e)}
        # =========================================================================

        # === 3.7) Quality Reflection (optional) =================================
        try:
            import os
            from services.quality_reflection import run_quality_reflection

            use_qr = str(os.getenv("USE_QUALITY_REFLECTION", "false")).lower() in (
                "1",
                "true",
                "yes",
            )
            if use_qr:
                qr = run_quality_reflection(
                    template_key=chosen_key,
                    spec=state.get("spec", {}),
                    py_src=py_src,
                    max_rounds=int(os.getenv("QUALITY_REFLECTION_ROUNDS", "1")),
                )
                py_src = qr.get("src", py_src)
                state["quality_reflection"] = {
                    "score": qr.get("score"),
                    "issues": qr.get("issues"),
                    "payloads": qr.get("payloads"),
                    "preflight_ok": qr.get("preflight_ok"),
                    "preflight_log": qr.get("preflight_log"),
                }
        except Exception as _e:
            state["quality_reflection"] = {"error": str(_e)}
        # =========================================================================

        # === 3.5) Reflection payloads injection + compile harmonization ===
        payloads = {}
        qref = state.get("quality_reflection")
        if isinstance(qref, dict):
            payloads = qref.get("payloads") or {}
        if not payloads:
            lg = state.get("langgraph_reflection")
            if isinstance(lg, dict):
                payloads = lg.get("slot_payloads") or {}

        if payloads:
            payloads = _alias_payload_keys(payloads)
            payloads = _normalize_compile_payload_vars(payloads)
            try:
                from services.llm_codegen_assist import apply_llm_slots

                py_src = apply_llm_slots(py_src, payloads)
            except Exception:
                pass
            py_src = _inject_slots_anystyle(py_src, payloads)

        # Compile argument alignment + sanitize + analyze (non-fatal)
        py_src = _ensure_compile_aligned(py_src)
        try:
            from services.quality_reflection import sanitize_code

            py_src = sanitize_code(py_src)
        except Exception:
            pass
        try:
            from services.code_quality_analyzer import analyze_quality

            aq = analyze_quality(py_src, spec)
            state["quality_after_compile_align"] = aq
        except Exception:
            pass

        # [ADDED] 새로운 아티팩트 필드 세팅
        state["basecode_py_path"] = py_path
        state["basecode_source"] = py_src
        state["basecode_summary"] = msum
        state["used_model"] = chosen_key  # 라우팅 결과 기록

        # (호환) 기존 base_code 필드도 유지
        state["base_code"] = py_src

        if "basecode_error" in state:
            del state["basecode_error"]

    except Exception as e:
        log.exception("base_code_node error")
        state["basecode_error"] = f"base_code_node error: {e}"

    return state


def build_graph():
    """
    그래프 구성:
      embedder → summary_node → classify_node → model_extractor → base_code
      (이후 user_input 있으면 qa_node, 없으면 종료)
    기존 흐름은 유지하고, spec/codegen 연동만 [CHANGED]/[ADDED] 지점으로 반영.
    """
    graph = StateGraph(AgentState)

    # 1) 노드 등록 (기존 구성 유지)
    graph.add_node("embedder", embedder)
    graph.add_node("summary_node", summarizer_agent)
    graph.add_node("classify_node", classifier_agent)

    # [ADDED] 추출/코드 생성 노드
    graph.add_node("model_extractor", model_extractor_node)  # [ADDED]
    graph.add_node("base_code_gen", base_code_node)  # [ADDED]

    graph.add_node("qa_node", qa_agent)

    # 2) 진입점 (기존 유지)
    graph.set_entry_point("embedder")

    # 3) 엣지 (기존 직렬 흐름 유지)
    graph.add_conditional_edges(
        "embedder", _should_qa_fast, {True: "qa_node", False: "summary_node"}
    )
    graph.add_edge("summary_node", "classify_node")
    graph.add_edge("classify_node", "model_extractor")  # [CHANGED] 항상 스펙 추출 실행
    graph.add_edge("model_extractor", "base_code_gen")  # [ADDED] codegen으로 연결

    # QA는 조건부 실행 (기존 로직 유지)
    def should_run_qa(state: AgentState) -> bool:
        return "user_input" in state and state["user_input"] is not None

    graph.add_conditional_edges(
        "base_code_gen",
        should_run_qa,
        {True: "qa_node", False: END},
    )

    graph.add_edge("qa_node", END)
    return graph.compile()
