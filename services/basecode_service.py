# services/basecode_service.py
# 목적: spec -> (generated .py, python source text, model summary) 아티팩트 반환
from typing import Dict, Any, Tuple
import io
from contextlib import redirect_stdout
import os
import json  # [ADDED]
import re
from pathlib import Path

from services import codegen
from services import llm_codegen_assist as lga  # [ADDED]
from services import template_registry  # [ADDED]
from services import routing  # [ADDED]


# [ADDED] 프로젝트 루트/영속 디렉토리
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # [ADDED]
PERSIST_DIR = os.path.join(PROJECT_ROOT, "uploaded_docs")  # [ADDED]
os.makedirs(PERSIST_DIR, exist_ok=True)


# [ADDED] caching utils
import hashlib, json, os, io
from contextlib import redirect_stdout


def _spec_fingerprint(spec: dict) -> str:
    """spec에서 비결정성 키 제거 후 안정적 해시."""
    safe = {k: v for k, v in (spec or {}).items() if k not in {"_evidence_texts"}}
    blob = json.dumps(safe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _stable_dir_for(doc_id: str) -> str:
    root = os.path.join("uploaded_docs", f"doc_{doc_id}")
    os.makedirs(root, exist_ok=True)
    return root


def _stable_artifact_path(doc_id: str, model_key: str, fp: str) -> str:
    d = _stable_dir_for(doc_id)
    return os.path.join(d, f"{model_key}_{fp}.py")


# [ADDED] codegen 헤더에서 메타 추출
_INFO_RE = re.compile(
    r"^\s*#\s*\[info\]\s*source_mode=([^\s]+)\s+template_key=([^\s]+)", re.M
)


def _extract_codegen_meta(py_src: str) -> Dict[str, Any]:
    """
    codegen.write_model_file()가 헤더로 남긴
    '# [info] source_mode=... template_key=...'를 파싱해 dict로 반환.
    """
    m = _INFO_RE.search(py_src or "")
    if m:
        return {"mode": m.group(1), "template_key": m.group(2)}
    return {}


def _persist_basecode(
    doc_id: str, model_key: str, py_src: str, model_summary: str
) -> str:
    """
    문서별 디렉토리에 base code 파일/요약/메타를 저장하고, py 파일 경로를 반환.
    """
    safe_id = str(doc_id).strip() or "noid"
    doc_dir = os.path.join(PERSIST_DIR, f"doc_{safe_id}")
    os.makedirs(doc_dir, exist_ok=True)

    py_path = os.path.join(doc_dir, f"{model_key}_basecode.py")
    sum_path = os.path.join(doc_dir, "basecode_summary.txt")
    meta_path = os.path.join(doc_dir, "basecode_meta.json")

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py_src)
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(model_summary)

    meta = {
        "doc_id": safe_id,
        "model_key": model_key,
        "py_path": py_path,
        "summary_path": sum_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return py_path


# [ADDED] LLM 보조 사용 토글 (원하면 False로 끌 수 있음)
USE_LLM_ASSIST: bool = True  # [ADDED]


def generate_base_code(model_key: str, spec: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Spec를 기반으로 베이스 코드를 생성하고, (py_path, py_source, model_summary)를 반환한다.
    RAW 모드(raw)일 경우: LLM 원본 코드를 파일로만 생성하고 import/compile/model.summary()를 수행하지 않는다.
    그 외 모드에서는 기존 동작(컴파일 및 model.summary 캡처)을 따른다.
    """
    # -------------------------------
    # 0) 캐시 키 및 모드 파라미터 결정
    # -------------------------------
    doc_id = str(
        spec.get("doc_id") or spec.get("document_id") or spec.get("id") or ""
    ).strip()
    fp = _spec_fingerprint(spec)
    prefer_cache = str(os.getenv("PREFER_STABLE", "1")).lower() in ("1", "true", "yes")

    # 하드닝/컴파일 모드: env > spec > 기본값
    _mode = (
        (
            str(
                os.getenv("BASECODE_MODE")
                or spec.get("hardening_mode")
                or "semantic_lite"
            )
        )
        .lower()
        .strip()
    )
    RAW_SUMMARY_MSG = (
        "<< RAW MODE >> Keras compile is skipped. "
        "The code below is LLM original without hardening."
    )

    # -------------------------------------------------
    # 1) 캐시 적중 시: raw 모드는 즉시 반환 (컴파일 시도 금지)
    # -------------------------------------------------
    if doc_id and prefer_cache:
        cached_py = _stable_artifact_path(doc_id, model_key, fp)
        if os.path.exists(cached_py):
            with open(cached_py, "r", encoding="utf-8") as f:
                py_source = f.read()

            if _mode == "raw":
                # RAW: 파일만 반환하고 요약은 안내문
                return cached_py, py_source, RAW_SUMMARY_MSG

            # RAW 외 모드에서는 캐시 파일을 임포트하여 summary 캡처
            import importlib.util, types

            spec_mod = importlib.util.spec_from_file_location(
                f"gen_cached_{model_key}_{fp}", cached_py
            )
            mod = importlib.util.module_from_spec(spec_mod)
            spec_mod.loader.exec_module(mod)  # noqa

            # build_model이 없을 수 있으므로 방어
            if not hasattr(mod, "build_model"):
                # 캐시는 있지만 모델 빌더가 없으면 일반 경로로 진행
                pass
            else:
                model = mod.build_model()  # noqa
                buf = io.StringIO()
                with redirect_stdout(buf):
                    try:
                        model.summary()
                    except Exception as e:
                        print(f"[WARN] model.summary() failed on cache: {e}")
                model_summary = buf.getvalue() or "<< EMPTY MODEL SUMMARY >>"
                return cached_py, py_source, model_summary

    # -----------------------------------
    # 2) spec 보강(선택) — 기존 동작 유지
    # -----------------------------------
    if USE_LLM_ASSIST:
        try:
            evidence_texts = spec.get("_evidence_texts") or []
            evidence_blob = "\n".join(evidence_texts)[:4000]
            patch = lga.propose_spec_patch(spec, evidence_blob)
            if getattr(patch, "patch", None):
                for k, v in patch.patch.items():
                    if k not in {"dangerous"}:
                        spec[k] = v
        except Exception:
            # 보조 실패는 무시 (안정성)
            pass

    # ---------------------------------------------------
    # 3) 파일 생성 + (raw)컴파일 스킵 / (그 외)컴파일 수행
    # ---------------------------------------------------
    model = None
    if _mode == "raw":
        # RAW: 렌더 후 파일만 생성
        py_path = codegen.write_model_file(model_key, spec)
    else:
        # 기존 동작: 파일 생성 → 임포트 → build_model() → compile
        model = codegen.build_compiled_model(model_key, spec)
        if model is None:
            raise RuntimeError(
                "build_compiled_model(...) returned None. "
                "Check that the generated module defines a builder (e.g., build_model/spec) "
                "and that pre-import guard didn't comment it out."
            )
        # codegen.build_compiled_model는 내부에서 write_model_file을 호출하므로 경로는 고정 규칙 사용
        py_path = f"{codegen.GENERATED_DIR}/{model_key}_generated.py"

    # ----------------------------------------
    # 4) 생성된 파이썬 소스 읽기 (공통)
    # ----------------------------------------
    with open(py_path, "r", encoding="utf-8") as f:
        py_source = f.read()

    # ----------------------------------------
    # 5) 모델 요약 캡처 (model이 있을 때만)
    # ----------------------------------------
    if model is not None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                model.summary()
            except Exception as e:
                print(f"[WARN] model.summary() failed (mode={_mode}): {e}")
        model_summary = buf.getvalue() or "<< EMPTY MODEL SUMMARY >>"
    else:
        # RAW 또는 빌더 미존재 등으로 model이 없는 경우
        model_summary = RAW_SUMMARY_MSG

    # --------------------------------------------------------
    # 6) 문서별 영속화(덮어쓰기 방지: doc_id별 별도 폴더) — 기존 유지
    # --------------------------------------------------------
    doc_id2 = str(
        spec.get("doc_id") or spec.get("document_id") or spec.get("id") or ""
    ).strip()
    if doc_id2:
        try:
            stable_py = _persist_basecode(doc_id2, model_key, py_source, model_summary)
            # UI에서 안전 경로를 쓰도록 py_path를 교체
            py_path = stable_py
        except Exception:
            # 저장 실패는 치명적이지 않으므로 무시
            pass

    return py_path, py_source, model_summary


# ============================================
# [ADDED] Transformer 강한 시그널 감지 → 강제 라우팅
# ============================================
def _force_transformer_if_signals(spec: Dict[str, Any]) -> bool:  # [ADDED]
    """
    다음 중 하나라도 만족하면 Transformer 확정:
      - proposed_model_family == 'transformer'
      - subtype in {'encoderdecoder', 'seq2seq'}
      - key_blocks에 'multiheadselfattention' 또는 'multi-head attention' 포함
      - task == machine_translation/sequence_to_sequence AND data_modality == text
    """
    family = str(spec.get("proposed_model_family") or "").strip().lower()
    subtype = str(spec.get("subtype") or "").strip().lower()
    task = str(spec.get("task_type") or "").strip().lower()
    modality = str(spec.get("data_modality") or "").strip().lower()

    kb = []
    try:
        kb_raw = spec.get("key_blocks") or []
        kb = [
            str(x).strip().lower()
            for x in (kb_raw if isinstance(kb_raw, (list, tuple)) else [])
        ]
    except Exception:
        kb = []

    if family == "transformer":
        return True
    if subtype in {"encoderdecoder", "seq2seq"}:
        return True
    if any(s in {"multiheadselfattention", "multi-head attention"} for s in kb):
        return True
    if task in {"machine_translation", "sequence_to_sequence"} and modality == "text":
        return True
    return False


# ============================================
# [MODIFIED] 간단 라우터: spec -> model_key
# - 선언형 라우팅(routing.resolve_template_from_spec) 우선
# - 강한 Transformer 시그널이면 무조건 'transformer'
# - 마지막 폴백은 'resnet'으로 완화(CNN family 오매칭 방지)
# ============================================
def decide_model_key_from_spec(spec: Dict[str, Any]) -> str:
    """
    1) 중앙 라우터(routing.resolve_template_from_spec) 우선
    2) 강한 Transformer 시그널이면 'transformer'
    3) 폴백: 'resnet' (과거 'cnn_family'로 잘못 가던 문제 완화)
    """
    # [ADDED] 중앙 라우터 우선
    try:
        template_key, meta = routing.resolve_template_from_spec(spec)  # [ADDED]
        if template_key:
            print("[routing]", (template_key, meta))  # [ADDED] 가시 로그
            return template_key
    except Exception:
        pass

    # [ADDED] 강한 Transformer 시그널 감지
    if _force_transformer_if_signals(spec):  # [ADDED]
        print("[routing] force → transformer (strong signals)")  # [ADDED]
        return "transformer"

    # [MODIFIED] 보수적 폴백: resnet (cnn_family 오매칭 방지)
    return "resnet"  # [MODIFIED]


# ============================================
# [ADDED] Auto wrapper
# - 외부에서 model_key를 몰라도 호출 가능
# - 내부에서 decide_model_key_from_spec()로 결정
# ============================================
# [ADDED] 레거시 키 → 실제 템플릿 파일 prefix 매핑
_TEMPLATE_ALIAS = {
    "cnn": "cnn_family",
    "vgg": "cnn_family",
    "mobilenet": "cnn_family",
    "efficientnet": "cnn_family",
    "inception": "cnn_family",
    "resnet": "resnet",  # 그대로 사용
    "transformer": "transformer",  # 그대로 사용
    "unet": "unet",
    "rnn": "rnn_seq",
    "lstm": "rnn_seq",
    "gru": "rnn_seq",
    "autoencoder": "autoencoder",
    "vae": "vae",
    "gan": "gan",
    "mlp": "mlp",
}


def generate_base_code_auto(spec: Dict[str, Any]) -> Tuple[str, str, str]:
    # [ADDED] 라우팅 우선
    model_key = None
    try:
        model_key, meta = routing.resolve_template_from_spec(spec)  # [ADDED]
        if model_key:
            print("[routing]", (model_key, meta))  # [ADDED]
    except Exception:
        model_key = None

    if not model_key:
        model_key = decide_model_key_from_spec(spec)  # 기존 폴백(이미 수정됨)

    # [ADDED] 최종 alias 정규화 (여기서 'cnn' → 'cnn_family' 강제)
    model_key_norm = (model_key or "").strip().lower()
    model_key = _TEMPLATE_ALIAS.get(model_key_norm, model_key_norm)

    # (선택) 디버깅 가시성
    print(f"[basecode] model_key(raw)={model_key_norm} -> template={model_key}")

    return generate_base_code(model_key, spec)
