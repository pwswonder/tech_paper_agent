"""
services/model_extractor_agent.py

Phase 1 (reboot): 논문 텍스트에서 '제안 모델' 스펙(ModelSpec)을 구조화 추출.
- GPT-4.1 (Azure) 사용 (openai_api_version="2024-10-21")
- 공유 템플릿에 맞춰 택소노미/힌트를 대폭 보강
- 출력은 엄격한 JSON → Pydantic(ModelSpec) 파싱 → spec_verifier로 검증/보정
"""

from __future__ import annotations
import os, json
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.spec_schema import ModelSpec

# === [ADDED] objective coercion helpers (lenient validation) ===
_ALLOWED_OBJECTIVES = {
    "mse",
    "mae",
    "cross_entropy",
    "binary_cross_entropy",
    "huber",
    "logcosh",
    "other",
}


def _coerce_objective(value: str) -> str:
    """Map free-text objective/loss to allowed literals; fallback to 'other'."""
    if not isinstance(value, str):
        return "other"
    s = value.strip().lower()
    if "binary" in s and ("cross entropy" in s or "cross-entropy" in s):
        return "binary_cross_entropy"
    if (
        "cross entropy" in s
        or "cross-entropy" in s
        or "nll" in s
        or "negative log" in s
    ):
        return "cross_entropy"
    if "mse" in s or "mean squared" in s:
        return "mse"
    if "mae" in s or "mean absolute" in s or "l1" in s:
        return "mae"
    if "huber" in s:
        return "huber"
    if "logcosh" in s or "log cosh" in s:
        return "logcosh"
    return "other"


# === [ADDED] family coercion helpers (lenient validation) ===
_ALLOWED_FAMILIES = {
    # 스키마에 등록된 문자열들(에러 메시지에서 본 값 포함)
    "Transformer",
    "transformer",
    "Transformer_MT",
    "transformer_mt",
    "TransformerMT",
    "transformermt",
    "Swin",
    "swin",
    "Performer",
    "performer",
    "Performer_MT",
    "performer_mt",
    "CNN",
    "cnn",
    "ResNet",
    "resnet",
    "VGG",
    "vgg",
    "DenseNet",
    "densenet",
    "RNN",
    "LSTM",
    "GRU",
    "GNN",
    "MLP",
    "Autoencoder",
    "VAE",
    "GAN",
    "Linear",
    "DLinear",
    "NLinear",
    "Other",
    "other",
}


def _coerce_family(value: str) -> str:
    """자유 텍스트 family를 스키마 허용 문자열로 보정. 모르면 'other'."""
    if not isinstance(value, str):
        return "other"
    s = value.strip()
    if s in _ALLOWED_FAMILIES:
        return s
    low = s.lower().replace("-", "_").replace(" ", "")
    # 소문자/별칭 매핑
    if low in {"transformer"}:
        return "transformer"
    if low in {"transformermt", "transformer_mt"}:
        return "transformer_mt"
    if low in {"swin"}:
        return "swin"
    if low in {"performer"}:
        return "performer"
    if low in {"performermt", "performer_mt"}:
        return "performer_mt"
    if low in {"cnn"}:
        return "cnn"
    if low in {"resnet"}:
        return "resnet"
    if low in {"vgg"}:
        return "vgg"
    if low in {"densenet"}:
        return "densenet"
    if low in {"rnn"}:
        return "RNN"
    if low in {"lstm"}:
        return "LSTM"
    if low in {"gru"}:
        return "GRU"
    if low in {"gnn"}:
        return "GNN"
    if low in {"mlp"}:
        return "MLP"
    if low in {"autoencoder"}:
        return "Autoencoder"
    if low in {"vae"}:
        return "VAE"
    if low in {"gan"}:
        return "GAN"
    if low in {"linear"}:
        return "Linear"
    if low in {"dlinear"}:
        return "DLinear"
    if low in {"nlinear"}:
        return "NLinear"
    # 예: diffusion 등 미등록 값 → 'other'
    return "other"


# === [ADDED] task_type coercion helpers (lenient validation) ===
_ALLOWED_TASK_TYPES = {
    "time_series_forecasting",
    "classification",
    "regression",
    "machine_translation",
    "text_summarization",
    "qa",
    "image_classification",
    "object_detection",
    "segmentation",
    "speech_recognition",
    "recommendation",
    "other",
}


def _coerce_task_type(value: str, spec: dict) -> str:
    """
    자유 텍스트 task_type을 스키마 허용값으로 보정.
    - 명확히 매핑되면 해당 값
    - 모호/미지원(예: image_translation)은 안전하게 'other'
    """
    if not isinstance(value, str):
        return "other"
    s = value.strip().lower().replace("-", " ").replace("_", " ")

    # 쉬운 1:1 매핑
    if "classification" in s:
        if "image" in s:
            return "image_classification"
        return "classification"
    if "regression" in s:
        return "regression"
    if "forecast" in s or "time series" in s:
        return "time_series_forecasting"
    if "summarization" in s or "summarisation" in s:
        return "text_summarization"
    if "question answering" in s or s == "qa" or "q a" in s:
        return "qa"
    if "object detection" in s or "detection" in s:
        return "object_detection"
    if "segment" in s:
        return "segmentation"
    if "speech" in s or "stt" in s or "speech to text" in s:
        return "speech_recognition"
    if "recommendation" in s or "recommender" in s or "ctr" in s:
        return "recommendation"

    # 'translation' 처리:
    # - 텍스트 번역(텍스트 포함) → machine_translation
    # - 이미지 번역(image-to-image style transfer 등)은 스키마에 없음 → 안전하게 'other'
    if "translation" in s or "translate" in s:
        modality = str(
            (spec or {}).get("modality") or (spec or {}).get("data_modality") or ""
        ).lower()
        if "text" in modality:
            return "machine_translation"
        # 텍스트 단서가 제목/요약 등 evidence에 있으면 텍스트 번역으로 판단
        blob = " ".join(
            str(x or "")
            for x in [
                (spec or {}).get("title"),
                (spec or {}).get("abstract"),
                (spec or {}).get("kw_blob"),
            ]
        ).lower()
        if "text" in blob or "sentence" in blob or "mt" in blob:
            return "machine_translation"
        # 그 외(image/image-to-image 등) → 스키마 미지원: other
        return "other"

    # 이미지 전용 태스크 (명시 키워드 없으면 other)
    if "image" in s and "classification" in s:
        return "image_classification"

    # 미등록/모호 값 → other
    return "other"


def _lenient_validate_spec(raw_json: dict):
    from pydantic_core import ValidationError as _PydValErr

    warnings = []

    # --- (1) objective 보정 (기존) ---
    obj = raw_json.get("objective")
    if isinstance(obj, str) and obj not in _ALLOWED_OBJECTIVES:
        coerced = _coerce_objective(obj)
        raw_json["objective"] = coerced if coerced in _ALLOWED_OBJECTIVES else "other"
        warnings.append(
            {"field": "objective", "raw": obj, "coerced": raw_json["objective"]}
        )
        raw_json["objective_detail"] = raw_json.get("objective_detail") or obj

    # --- [ADDED] (2) family 보정 (top-level + baselines.*.family) ---
    # top-level family (있다면)
    fam = raw_json.get("family")
    if isinstance(fam, str) and fam not in _ALLOWED_FAMILIES:
        coerced = _coerce_family(fam)
        raw_json["family"] = coerced
        warnings.append({"field": "family", "raw": fam, "coerced": coerced})
        # 상세 보존
        if coerced == "other":
            raw_json["family_detail"] = raw_json.get("family_detail") or fam

    # baselines 배열 내부의 family
    bl = raw_json.get("baselines")
    if isinstance(bl, list):
        for idx, b in enumerate(bl):
            if not isinstance(b, dict):
                continue
            bf = b.get("family")
            if isinstance(bf, str) and bf not in _ALLOWED_FAMILIES:
                coerced = _coerce_family(bf)
                b["family"] = coerced
                warnings.append(
                    {"field": f"baselines[{idx}].family", "raw": bf, "coerced": coerced}
                )
                if coerced == "other":
                    b["family_detail"] = b.get("family_detail") or bf

    # --- [ADDED] (3) task_type 보정 ---
    tt = raw_json.get("task_type")
    if isinstance(tt, str) and tt not in _ALLOWED_TASK_TYPES:
        coerced = _coerce_task_type(tt, raw_json)
        raw_json["task_type"] = coerced
        warnings.append({"field": "task_type", "raw": tt, "coerced": coerced})
        if coerced == "other":
            # 원문 보존
            raw_json["task_type_detail"] = raw_json.get("task_type_detail") or tt

    # baselines 내부 task_type도 있으면 보정
    bl = raw_json.get("baselines")
    if isinstance(bl, list):
        for idx, b in enumerate(bl):
            if not isinstance(b, dict):
                continue
            btt = b.get("task_type")
            if isinstance(btt, str) and btt not in _ALLOWED_TASK_TYPES:
                c = _coerce_task_type(btt, raw_json)
                b["task_type"] = c
                warnings.append(
                    {"field": f"baselines[{idx}].task_type", "raw": btt, "coerced": c}
                )
                if c == "other":
                    b["task_type_detail"] = b.get("task_type_detail") or btt

    # 검증 재시도
    try:
        model = ModelSpec.model_validate(raw_json)
        return model, warnings
    except _PydValErr:
        # 최후의 수단: 남은 family류 값을 'other'로 강제
        if isinstance(bl, list):
            for b in bl:
                if (
                    isinstance(b, dict)
                    and "family" in b
                    and b["family"] not in _ALLOWED_FAMILIES
                ):
                    b["family"] = "other"
        if (
            isinstance(raw_json.get("family"), str)
            and raw_json["family"] not in _ALLOWED_FAMILIES
        ):
            raw_json["family"] = "other"

            if isinstance(bl, list):
                for b in bl:
                    if (
                        isinstance(b, dict)
                        and "task_type" in b
                        and b["task_type"] not in _ALLOWED_TASK_TYPES
                    ):
                        b["task_type"] = "other"
            if (
                isinstance(raw_json.get("task_type"), str)
                and raw_json["task_type"] not in _ALLOWED_TASK_TYPES
            ):
                raw_json["task_type"] = "other"

        model = ModelSpec.model_validate(raw_json)
        warnings.append({"field": "<fallback>", "raw": "<family>", "coerced": "other"})
        return model, warnings


# === [END ADDED] ===

from services.spec_verifier import verify_and_normalize

# ------------------------------------------------------------
# 0) LLM 설정 (GPT-4.1)
# ------------------------------------------------------------
load_dotenv()
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
    openai_api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.0,
)

# ------------------------------------------------------------
# 1) 프롬프트 (proposed-only + 새 택소노미 힌트)
# ------------------------------------------------------------
SYSTEM = """You are an expert research paper extractor.
CRITICAL: Extract ONLY the PROPOSED model of the paper (NOT baselines).
Return STRICT JSON following the provided schema example. If uncertain, set
`confidence` low and `is_proposed_clearly_identified=false`. Include short
evidence snippets that justify the family/subtype choice."""

HINTS = """
Examples:
- Families: ["Transformer","CNN","RNN","LSTM","GRU","GNN","MLP","Autoencoder","VAE","GAN","Linear","DLinear","NLinear","Other"].
- CNN subtypes (examples): ["ResNet","VGG","DenseNet","Inception","MobileNet","UNet","Conv1D","Conv2D"].
- Transformer subtypes: ["Encoder","Decoder","EncoderDecoder"].
- RNN subtypes: ["Seq2Seq","Sequence"].
- For GAN/VAE/Autoencoder, set family accordingly and use key_blocks like ["Conv2D","Conv2DTranspose","Sampling","KL","Dense"] as appropriate.
- task_type ∈ ["time_series_forecasting","classification","regression","machine_translation","text_summarization","qa","image_classification","object_detection","segmentation","speech_recognition","recommendation","other"].
- data_modality ∈ ["time_series","text","image","audio","tabular","graph","multimodal","other"].
- positional_encoding ∈ ["absolute","relative","none","other"].
"""

SCHEMA_EXAMPLE = r"""
{
  "title": "<string or null>",
  "task_type": "<TaskType>",
  "data_modality": "<DataModality>",
  "proposed_model_family": "<ModelFamily>",
  "subtype": "<string or null>",
  "key_blocks": ["<string>", "..."],
  "dims": {
    "in_dim": <int or null>,
    "out_dim": <int or null>,
    "seq_len": <int or null>,
    "pred_len": <int or null>,
    "hidden_dim": <int or null>,
    "num_layers": <int or null>,
    "num_heads": <int or null>,
    "ffn_dim": <int or null>,
    "dropout": <float or null>,
    "height": <int or null>,
    "width": <int or null>,
    "kernel_size": <int or null>,
    "dilation": <int or null>,
    "vocab_size": <int or null>,
    "max_len": <int or null>
  },
  "objective": "<ObjectiveType or null>",
  "positional_encoding": "<PositionalEncodingType or null>",
  "baselines": [
    {"name": "<string>", "family": "<ModelFamily or null>", "notes": "<string or null>"}
  ],
  "evidence": [
    {"text": "<short quote>", "section": "<string or null>", "page": <int or null>}
  ],
  "confidence": <float between 0 and 1>,
  "is_proposed_clearly_identified": <true|false>
}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "user",
            """Extract the PROPOSED model SPEC from the paper content below.

# HINTS
{hints}

# REQUIRED JSON SCHEMA (EXAMPLE FORMAT)
{json_schema}

# PAPER CONTENT
Title (if known): {title}
---
{paper_text}
""",
        ),
    ]
)

_parser = StrOutputParser()


# ------------------------------------------------------------
# 2) JSON 안전 파싱
# ------------------------------------------------------------
def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    b, e = s.find("{"), s.rfind("}")
    if b != -1 and e != -1 and e > b:
        s = s[b : e + 1]
    return json.loads(s)


# ------------------------------------------------------------
# 3) 외부 인터페이스
# ------------------------------------------------------------
def extract_model_spec(
    paper_text: str, title: Optional[str] = None, strict: bool = True
) -> Dict[str, Any]:
    """
    입력: 논문 본문/요약 텍스트
    출력: {"raw": <ModelSpec dict>, "verified": <VerifiedSpec dict>, "warnings": [..]}
    - raw: LLM이 생성한 JSON을 Pydantic(ModelSpec)으로 파싱
    - verified: 검증/보정 결과
    - warnings: 엄격 검증 실패 시 보정(coercion) 내역
    """
    # 1) LLM 체인 실행 (기존 로직 유지)
    chain = PROMPT | llm | _parser
    out = chain.invoke(
        {
            "hints": HINTS,
            "json_schema": SCHEMA_EXAMPLE,
            "paper_text": paper_text[:40000],  # 과도한 길이 방지
            "title": title or "(unknown)",
        }
    )

    # 2) JSON 파싱 (기존 로직 유지)
    raw_json = _safe_json_loads(out)

    # 3) 검증: strict → 실패 시 lenient 보정 (objective 등)
    warnings: list = []
    if strict:
        try:
            # 엄격 검증 시도
            raw_spec = ModelSpec.model_validate(raw_json)
        except Exception:
            # 완화 경로: objective 등 자유 텍스트를 허용 리터럴로 보정
            raw_spec, lw = _lenient_validate_spec(raw_json)
            if isinstance(lw, list):
                warnings.extend(lw)
    else:
        # strict=False인 경우 처음부터 완화 경로
        raw_spec, lw = _lenient_validate_spec(raw_json)
        if isinstance(lw, list):
            warnings.extend(lw)

    # 4) 사양 검증/보정 (기존 로직 유지)
    verified = verify_and_normalize(raw_spec)

    # 5) 반환 (raw/verified + warnings)
    return {
        "raw": json.loads(raw_spec.model_dump_json()),
        "verified": json.loads(verified.model_dump_json()),
        "warnings": warnings,
    }


if __name__ == "__main__":
    # 간단 데모: CNN(ResNet) 느낌의 텍스트
    demo = """
    We propose a residual CNN architecture for image classification on CIFAR-10.
    Our model stacks multiple residual blocks with Conv2D layers and uses global average pooling.
    """
    res = extract_model_spec(demo, title="Residual CNN for CIFAR")
    print(json.dumps(res, indent=2, ensure_ascii=False))
