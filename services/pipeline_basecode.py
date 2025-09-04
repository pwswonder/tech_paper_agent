# -*- coding: utf-8 -*-
"""
services/pipeline_basecode.py

단일 오케스트레이터(최종 정렬판)
- spec → 템플릿 선택 → 렌더 → 슬롯 채움(LLM→autoblock) → sanitize/품질 → 리플렉션
  → compile 하모나이저 → 최종 봉합 → 영속화

주요 포인트
1) 타입 안전: 검증기는 ModelSpec(Pydantic) 기대 → 검증 이후 dict로 변환해 처리
2) 슬롯 정책: 초기에는 *보존*(payload 없으면 치환하지 않음). 모든 주입 단계가 끝난 다음에만 남은 슬롯을 pass로 봉합
3) any-style 슬롯 주입기: 주석/RAW/Jinja/단독 중괄호 모두 탐지/치환
4) alias 매핑: model_head→imports_extra, encoder/decoder(_blocks)→encoder_layers/decoder_layers, compile→compile_override 등
5) compile 하모나이저: 템플릿의 `model.compile(...)`이 항상
   `optimizer=optimizer, loss=loss_fn, metrics=metrics`를 사용하도록 자동 정렬
6) 변수명 정합: 리플렉션/LLM payload가 `opt = ...` 를 만들더라도, 주입 직전에 `optimizer = ...`로 정규화

환경 변수
- USE_LLM_ASSIST=true|false
- USE_QUALITY_REFLECTION=true|false
- DEBUG_BASECODE=true|false
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Set
import os, re, json, io

# -------------------------
# 안전 임포트(services.* 우선)
# -------------------------
from services.spec_schema import ModelSpec
from services.spec_verifier import verify_and_normalize
from services.basecode_service import decide_model_key_from_spec, _persist_basecode
from services.spec_hardener import harden_spec_for_template
from services.codegen import render_model_source, write_model_file
from services.llm_codegen_assist import apply_llm_slots
from services.codegen_autoblocks import autofill_custom_blocks
from services.code_quality_analyzer import analyze_quality
from services.quality_reflection import run_quality_reflection, sanitize_code
from services.langgraph_reflection import run_langgraph_reflection
from services.spec_schema import ModelSpec  # Pydantic 스키마


# -------------------------
# 설정
# -------------------------
USE_LLM = str(os.getenv("USE_LLM_ASSIST", "true")).lower() in ("1", "true", "yes")
USE_REFLECTION = str(os.getenv("USE_QUALITY_REFLECTION", "true")).lower() in (
    "1",
    "true",
    "yes",
)
DEBUG = str(os.getenv("DEBUG_BASECODE", "false")).lower() in ("1", "true", "yes")


def _debug(*args):
    if DEBUG:
        print("[pipeline_basecode]", *args)


# -------------------------
# 유틸: 타입 정규화 / 슬롯 처리
# -------------------------
def _as_plain_dict(x: Any) -> Dict[str, Any]:
    """dict/pydantic/tuple 등을 안전하게 dict로 변환."""
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            pass
    try:
        return dict(x)  # 마지막 폴백
    except Exception as e:
        raise TypeError(f"_as_plain_dict: unsupported type {type(x)}: {e}")


# 다양한 스타일의 CUSTOM_BLOCK 라인(Jinja RAW/주석/단독 중괄호 포함)을 한 번에 탐지
_SLOT_ANY_RE = re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?:#\s*)?"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"
    r"(?:\{\{CUSTOM_BLOCK:\s*(?P<name1>[A-Za-z0-9_\-]+)\s*\}\}|\{CUSTOM_BLOCK:\s*(?P<name2>[A-Za-z0-9_\-]+)\s*\})"
    r"(?:\s*\{\%\s*endraw\s*\%\})?\s*$"
)


def _find_slots_anystyle(text: str) -> Set[str]:
    names: Set[str] = set()
    for m in _SLOT_ANY_RE.finditer(text or ""):
        name = (m.group("name1") or m.group("name2") or "").strip()
        if name:
            names.add(name)
    return names


def _inject_slots_anystyle(text: str, payloads: Dict[str, str]) -> str:
    """
    다양한 스타일의 슬롯 라인을 코드 블록으로 치환.
    payload가 없으면 *보존*(치환하지 않음)하여, 후속 단계(리플렉션)에서 주입될 기회를 남긴다.
    """
    if not isinstance(payloads, dict) or not payloads:
        return text

    def _repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        name = (m.group("name1") or m.group("name2") or "").strip()
        repl = payloads.get(name)
        if repl is None or repl == "":
            # 보존: 리플렉션/후단에서 주입 가능하도록 원문 유지
            return m.group(0)
        lines = (repl or "").splitlines()
        return "\n".join(indent + ln for ln in lines if ln is not None)

    return _SLOT_ANY_RE.sub(_repl, text)


def _finalize_leftover_slots_to_pass(text: str) -> str:
    """
    파이프라인 마지막 단계에서만 호출: 남은 슬롯을 컴파일 가능한 pass로 봉합.
    원문 마커를 주석으로 남겨 추후 디버깅/후수정에 도움을 준다.
    """

    def _repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        name = (m.group("name1") or m.group("name2") or "").strip()
        return f"{indent}pass  # {{CUSTOM_BLOCK:{name}}}"

    return _SLOT_ANY_RE.sub(_repl, text)


# 리플렉션/외부 모듈에서 들어오는 슬롯 키를 템플릿의 실제 슬롯 명으로 매핑
SLOT_ALIASES = {
    # 공통
    "model_head": "imports_extra",
    "imports": "imports_extra",
    "compile": "compile_override",
    # MT 자주 쓰는 별칭
    "encoder": "encoder_layers",
    "encoder_blocks": "encoder_layers",
    "decoder": "decoder_layers",
    "decoder_blocks": "decoder_layers",
}


def _alias_payload_keys(payloads: Dict[str, str]) -> Dict[str, str]:
    if not isinstance(payloads, dict):
        return {}
    out = {}
    for k, v in payloads.items():
        out[SLOT_ALIASES.get(k, k)] = v
    return out


# -------------------------
# compile 변수/호출 하모나이저
# -------------------------
def _normalize_compile_payload_vars(payloads: Dict[str, str]) -> Dict[str, str]:
    """
    payloads['compile_override'] 내부의 변수명을 파이프라인/템플릿 규약에 맞추어 정규화:
    - opt = ...  → optimizer = ...
    - optimizer 변수명은 'optimizer'로 고정
    - loss 변수명은 'loss_fn', metrics 변수명은 'metrics'로 사용
    """
    if not isinstance(payloads, dict):
        return {}
    text = payloads.get("compile_override")
    if not isinstance(text, str):
        return payloads

    # 1) 'opt =' → 'optimizer =' 로 정규화
    text2 = re.sub(r"(?m)^\s*opt\s*=", "optimizer = ", text)

    # 2) 안전한 줄바꿈 보정:
    #   - '...optimizer = ...loss_fn' 같이 붙은 경우 분리
    #   - '...)metrics' 같이 붙은 경우 분리
    # (a) loss_fn 앞에 \n 보장
    text2 = re.sub(r"(?<!\n)(loss_fn\s*=\s*)", r"\n\1", text2)
    # (b) metrics 앞에 \n 보장
    text2 = re.sub(r"(?<!\n)(metrics\s*=\s*)", r"\n\1", text2)

    # 3) 양쪽 공백 정리(선택)
    text2 = re.sub(r"[ \t]+\n", "\n", text2)  # 라인말 공백 제거

    # 2) 정규화한 텍스트를 반영
    out = dict(payloads)
    out["compile_override"] = text2
    return out


def _rewrite_compile_call_to_spec_vars(py_src: str) -> str:
    """
    소스 내 첫 번째 model.compile(...) 호출의 인자를
    'optimizer=optimizer, loss=loss_fn, metrics=metrics' 로 교체.
    - 괄호 균형을 간단 파서로 맞춰 안전 치환
    - 이미 원하는 형태면 그대로 반환
    """
    target = "model.compile("
    i = py_src.find(target)
    if i == -1:
        return py_src  # compile 호출 없음
    around = py_src[max(0, i - 120) : i + 240]
    if (
        ("optimizer=optimizer" in around)
        and ("loss=loss_fn" in around)
        and ("metrics=metrics" in around)
    ):
        return py_src

    # 괄호 균형으로 닫힘 위치 찾기
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
    if depth != 0:
        return py_src  # 비정상

    before = py_src[:start_args]
    after = py_src[j - 1 :]  # 닫는 ')' 포함 앞부분은 교체
    replacement = "optimizer=optimizer, loss=loss_fn, metrics=metrics"
    return before + replacement + after


def _ensure_compile_aligned(py_src: str) -> str:
    """
    - compile 호출이 있으면 인자를 spec 변수 기반으로 치환:
      optimizer=optimizer, loss=loss_fn, metrics=metrics
    - compile 호출이 없으면 'return model' 직전에 model.compile(...) 삽입
    """
    if "model.compile(" in py_src:
        return _rewrite_compile_call_to_spec_vars(py_src)

    # compile 호출이 없다면 'return model' 직전에 삽입
    lines = py_src.splitlines()
    inserted = False
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().startswith("return model"):
            indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
            ins = (
                indent
                + "model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)"
            )
            lines.insert(idx, ins)
            inserted = True
            break
    if not inserted:
        lines.append("")
        lines.append(
            "model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)"
        )
    return "\n".join(lines)


# -------------------------
# 메인 오케스트레이터
# -------------------------
def generate_high_quality_basecode(
    raw_spec: Dict[str, Any],
    *,
    doc_id: Optional[str] = None,
    max_reflect_rounds: int = 2,
    out_dir: Optional[str] = None,
    override_template_key: Optional[
        str
    ] = None,  # 테스트 편의를 위한 옵션(서비스 기본: None)
) -> Dict[str, Any]:
    """
    입력:
      - raw_spec: 업로드 문서에서 추출된 spec(dict 또는 ModelSpec)
      - doc_id: 저장 경로 분기용 식별자
      - max_reflect_rounds: 리플렉션 반복 횟수(보수적으로 2)
      - out_dir: None이면 codegen 기본 디렉토리 사용
      - override_template_key: (테스트 전용) 라우팅 무시하고 템플릿 강제

    반환:
      {
        "template_key": str,
        "py_path": str,
        "py_src": str,
        "summary": str,
        "quality": dict,
        "reflection": dict,
        "slots_used": dict,
      }
    """
    # 1) 스펙 검증/정규화 (검증기 기대 타입: ModelSpec)
    mspec = (
        raw_spec if isinstance(raw_spec, ModelSpec) else ModelSpec(**(raw_spec or {}))
    )
    v = verify_and_normalize(mspec)

    # 이후 단계 편의를 위해 dict로 변환
    try:
        spec = v.spec.model_dump()
    except Exception:
        try:
            spec = v.spec.dict()
        except Exception:
            spec = _as_plain_dict(v.spec)

    if doc_id:
        spec["doc_id"] = doc_id

    # 2) 템플릿 선택(라우터 우선) + (옵션) override
    if override_template_key:
        template_key = override_template_key.strip().lower()
        _debug("routing (override) →", template_key)
    else:
        template_key = decide_model_key_from_spec(spec)
        _debug("routing →", template_key)

    # 3) 템플릿별 하드닝 (반환 타입 가변성에 안전)
    hs = harden_spec_for_template(spec, template_key)
    updated_template_key = None
    if isinstance(hs, tuple):
        candidate_spec = hs[0]
        maybe_aux = hs[1] if len(hs) > 1 else None
        if isinstance(maybe_aux, str) and maybe_aux.strip():
            updated_template_key = maybe_aux.strip().lower()
        spec = _as_plain_dict(candidate_spec)
    else:
        spec = _as_plain_dict(hs)

    if updated_template_key:
        template_key = updated_template_key

    # dims 평탄화: 루트에 없으면만 채움
    dims = spec.get("dims") or {}
    if isinstance(dims, dict):
        for k, v in dims.items():
            spec.setdefault(k, v)

    # 4) 1차 렌더
    py_src = render_model_source(template_key, spec)

    # 5) 슬롯 채움 (LLM → autoblocks → 리플렉션 → compile 하모나이저 → 최종 봉합)
    slots_used: Dict[str, str] = {}

    # 5-1) LLM 슬롯 주입 (spec.custom_blocks → alias → compile 변수 정규화 → any-style)
    if USE_LLM:
        try:
            payloads = _alias_payload_keys(spec.get("custom_blocks") or {})
            payloads = _normalize_compile_payload_vars(
                payloads
            )  # ✅ opt → optimizer 정규화
            if payloads:
                try:
                    py_src = apply_llm_slots(py_src, payloads)
                except Exception:
                    pass
                py_src = _inject_slots_anystyle(py_src, payloads)
                slots_used.update(payloads)
                _debug("LLM slots injected:", list(payloads.keys()))
        except Exception as e:
            _debug("LLM slot injection skipped due to:", e)

    # 5-2) 남은 슬롯 → autoblocks 보강 → 재렌더 → 재주입
    leftover_before = _find_slots_anystyle(py_src)
    if leftover_before:
        _debug("leftover before autoblock:", leftover_before)
        try:
            family = (
                spec.get("family") or spec.get("model_family") or ""
            ).lower() or None
            spec = autofill_custom_blocks(spec, family=family)
            py_src = render_model_source(template_key, spec)
            payloads2 = _alias_payload_keys(spec.get("custom_blocks") or {})
            payloads2 = _normalize_compile_payload_vars(payloads2)  # ✅ opt → optimizer
            if payloads2:
                try:
                    py_src = apply_llm_slots(py_src, payloads2)
                except Exception:
                    pass
                py_src = _inject_slots_anystyle(py_src, payloads2)
                slots_used.update(payloads2)
                _debug("autoblocks slots injected:", list(payloads2.keys()))
        except Exception as e:
            _debug("autoblocks failed:", e)

    # 6) sanitize + 품질 분석
    py_src = sanitize_code(py_src)
    quality = analyze_quality(py_src, spec)

    # 7) 리플렉션 루프
    reflection = {
        "src": py_src,
        "score": quality.get("score", 0),
        "issues": quality.get("issues", []),
    }
    if USE_REFLECTION:
        try:
            ref = run_quality_reflection(
                template_key, spec, py_src, enable=True, max_rounds=max_reflect_rounds
            )
            if isinstance(ref, dict) and ref.get("src"):
                reflection = ref
                py_src = ref["src"]
        except Exception as e:
            _debug("quality_reflection failed:", e)
            try:
                ref2 = run_langgraph_reflection(
                    py_src, spec, max_rounds=max_reflect_rounds
                )
                if isinstance(ref2, dict) and ref2.get("src"):
                    reflection = ref2
                    py_src = ref2["src"]
            except Exception as e2:
                _debug("langgraph_reflection failed:", e2)

    # 7.1) 리플렉션 payloads → alias → compile 변수 정규화 → any-style 주입
    try:
        payloads_ref = _alias_payload_keys(reflection.get("payloads") or {})
        payloads_ref = _normalize_compile_payload_vars(
            payloads_ref
        )  # ✅ opt → optimizer
        if payloads_ref:
            try:
                py_src = apply_llm_slots(py_src, payloads_ref)
            except Exception:
                pass
            py_src = _inject_slots_anystyle(py_src, payloads_ref)
            slots_used.update(payloads_ref)
            py_src = sanitize_code(py_src)
            quality = analyze_quality(py_src, spec)
            _debug("reflection slots injected:", list(payloads_ref.keys()))
    except Exception as e:
        _debug("reflection payload injection failed:", e)

    # 7.2) compile 하모나이저로 최종 정렬
    py_src = _ensure_compile_aligned(py_src)
    py_src = sanitize_code(py_src)
    quality = analyze_quality(py_src, spec)

    # 7.3) 남아있는 슬롯을 pass로 봉합(마커 보존)
    final_leftover = _find_slots_anystyle(py_src)
    if final_leftover:
        _debug("final leftover before pass seal:", final_leftover)
        py_src = _finalize_leftover_slots_to_pass(py_src)
        py_src = sanitize_code(py_src)
        quality = analyze_quality(py_src, spec)

    # 8) 영속화
    py_path = ""
    try:
        if doc_id:
            py_path = _persist_basecode(
                doc_id, template_key, py_src, quality.get("summary", "")
            )
        else:
            py_path = write_model_file(template_key, spec, out_dir or None)
    except Exception as e:
        _debug("persistence failed:", e)
        py_path = ""

    return {
        "template_key": template_key,
        "py_path": py_path,
        "py_src": py_src,
        "summary": quality.get("summary", ""),
        "quality": quality,
        "reflection": reflection,
        "slots_used": slots_used,
    }
