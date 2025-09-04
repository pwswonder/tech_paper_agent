"""
services/spec_verifier.py

Phase 1 (reboot): 공유 템플릿 군을 고려한 '범용' 검증기.
- 공통 검증(제안모델 식별, 모달리티-태스크 일관성)
- family별 '핵심 블록' 시그니처 존재 여부 점검(경고 위주)
- 시계열 Linear류(선택) 등 가벼운 소프트 보정
- 과잉 보정은 지양(항상 증거와 스펙이 우선)
"""

from __future__ import annotations
from typing import List, Dict, Set, Callable
from services.spec_schema import ModelSpec, VerifiedSpec, VerificationWarning

# ------------------------------------------------------------
# 0) family별 '핵심 블록' 시그니처 (템플릿들과 합치되는 최소 기준)
#    - 경고에만 사용(강제 X). Self-Check(Phase3)에서 실제 코드 검증을 수행.
# ------------------------------------------------------------
REQUIRED_BLOCKS_BY_FAMILY: Dict[str, Set[str]] = {
    "transformer": {"multiheadattention", "layernormalization"},
    "cnn": {"conv2d"},  # 1D의 경우 conv1d는 템플릿 메타에서 처리
    "lstm": {"lstm"},
    "gru": {"gru"},
    "rnn": {"rnn"},  # 거의 사용 적음(호환용)
    "gnn": {"gcn", "graphconv", "gat"},  # 구현에 따라 OR로 만족
    "mlp": {"dense"},
    "autoencoder": {"dense"},  # conv 기반 AE는 conv2d가 key_blocks에 포함될 것
    "vae": {"dense"},  # 간이 기준(실전은 reparameterization/kl 추가)
    "gan": {"conv2dtranspose", "dense"},  # DCGAN류 힌트
    "dlinear": {"linear", "moving average"},
    "nlinear": {"linear"},
    "linear": {"linear"},
}

OPTIONAL_BLOCKS_BY_FAMILY: Dict[str, Set[str]] = {
    "transformer": {"residual", "feedforward", "positionalencoding", "crossattention"},
    "cnn": {
        "batchnormalization",
        "globalaveragepooling2d",
        "maxpooling2d",
        "upsampling2d",
    },
    "gan": {"generator", "discriminator"},
    "autoencoder": {"conv2d", "conv2dtranspose"},
    "vae": {"sampling", "kl"},
}

# family별 추가 검증 함수(선택)
FamilyValidator = Callable[[ModelSpec, List[VerificationWarning]], None]
FAMILY_VALIDATORS: Dict[str, FamilyValidator] = {}


def register_family_validator(family: str):
    def _wrap(fn: FamilyValidator):
        FAMILY_VALIDATORS[family.lower()] = fn
        return fn

    return _wrap


@register_family_validator("Transformer")
def _validate_transformer(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    blocks = {b.lower() for b in spec.key_blocks}
    if not any("multiheadattention" in b for b in blocks):
        warns.append(
            VerificationWarning(
                code="TRANSFORMER_MHA_MISSING",
                message="Transformer인데 MultiHeadAttention 단서가 희박합니다.",
                fix_applied=False,
            )
        )


@register_family_validator("GAN")
def _validate_gan(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    # GAN은 학습 루틴/두 네트워크 존재 등이 중요하지만, 여기선 시그니처만 가볍게 안내
    warns.append(
        VerificationWarning(
            code="GAN_TRAINING_NOTE",
            message="GAN은 generator/discriminator 두 네트워크와 적대적 학습 루틴이 필요합니다(Phase3 학습루틴에서 점검 권장).",
            fix_applied=False,
        )
    )


# ------------------------------------------------------------
# 1) 공통 검증 & 소프트 보정
# ------------------------------------------------------------
_TS_HINTS = {"time series", "forecast", "long-term forecasting", "temporal", "seq2pred"}
_LINEAR_HINTS = {"dlinear", "nlinear", "linear", "moving average", "decomposition"}


def _common_checks(spec: ModelSpec) -> List[VerificationWarning]:
    warns: List[VerificationWarning] = []
    if not spec.is_proposed_clearly_identified:
        warns.append(
            VerificationWarning(
                code="PROPOSED_UNCLEAR",
                message="제안 모델 식별이 불명확합니다. 증거/본문 재확인 권장.",
                fix_applied=False,
            )
        )
    if spec.data_modality == "time_series" and spec.task_type not in (
        "time_series_forecasting",
        "regression",
        "other",
    ):
        warns.append(
            VerificationWarning(
                code="MODALITY_TASK_MISMATCH",
                message=f"time_series 모달리티인데 task_type={spec.task_type}. 재확인 권장.",
                fix_applied=False,
            )
        )

    fam = (spec.proposed_model_family or "Other").lower()
    req = REQUIRED_BLOCKS_BY_FAMILY.get(fam, set())
    blocks = {b.lower() for b in spec.key_blocks}
    if req and not any(any(r in b for r in req) for b in blocks):
        warns.append(
            VerificationWarning(
                code="REQUIRED_BLOCKS_WEAK",
                message=f"{spec.proposed_model_family} 핵심 블록 단서가 희박합니다(요구 힌트: {sorted(list(req))}).",
                fix_applied=False,
            )
        )
    return warns


def _soft_coercions(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    # 아주 보수적으로만 작동
    ev = " ".join([e.text.lower() for e in spec.evidence])
    if spec.data_modality == "time_series" and (
        any(k in ev for k in _LINEAR_HINTS) or any(k in ev for k in _TS_HINTS)
    ):
        if spec.proposed_model_family in ("Other", "MLP", "RNN", "LSTM", "GRU"):
            spec.proposed_model_family = "DLinear"
            if not spec.subtype:
                spec.subtype = "DLinear"
            warns.append(
                VerificationWarning(
                    code="COERCE_TO_DLINEAR",
                    message="시계열+Linear 단서가 강해 family를 DLinear로 보정했습니다.",
                    fix_applied=True,
                )
            )

    # [NEW] 제안 모델 명시성 보정: family/dims/blocks/evidence 단서가 있으면 표시
    try:
        if not spec.is_proposed_clearly_identified:
            fam = (spec.proposed_model_family or "Other").lower()
            # dims가 비어 있지 않으면 강한 단서로 간주(스키마 확장과 무관하게 동작)
            has_struct_dims = False
            dims = getattr(spec, "dims", None)
            if dims is not None:
                try:
                    dmp = dims.model_dump(exclude_none=True)
                    has_struct_dims = bool(dmp)
                except Exception:
                    has_struct_dims = False
            has_blocks = bool(getattr(spec, "key_blocks", None))
            has_evidence = bool(getattr(spec, "evidence", None))

            known = {
                "transformer",
                "transformer_mt",
                "swin",
                "performer",
                "performer_mt",
                "cnn",
                "resnet",
                "vgg",
                "densenet",
                "rnn",
                "lstm",
                "gru",
                "gnn",
                "mlp",
                "autoencoder",
                "vae",
                "gan",
                "linear",
                "dlinear",
                "nlinear",
                "other",
            }
            if fam in known and (has_struct_dims or has_blocks or has_evidence):
                spec.is_proposed_clearly_identified = True
                # 기존 PROPOSED_UNCLEAR 경고가 이미 추가되었다면 제거
                for i in range(len(warns) - 1, -1, -1):
                    if getattr(warns[i], "code", "") == "PROPOSED_UNCLEAR":
                        warns.pop(i)
                warns.append(
                    VerificationWarning(
                        code="PROPOSED_MARKED_CLEAR",
                        message="family/dims/blocks/evidence 단서를 근거로 제안 모델을 명시적으로 표시했습니다.",
                        fix_applied=True,
                    )
                )
    except Exception:
        # 절대 검증을 깨뜨리지 않음
        pass


# ===================== NEW VALIDATORS (session extensions) =====================
from typing import Any

try:
    FAMILY_VALIDATORS  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    FAMILY_VALIDATORS = {}  # minimal fallback if not declared above

FamilyValidator = Callable[[ModelSpec, List[VerificationWarning]], None]


@register_family_validator("swin")
def _validate_swin(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    # Minimal structural expectations for Swin-like backbone
    dims = spec.dims or type(spec).model_fields["dims"].default_factory()
    sw = getattr(dims, "swin", None)
    ok = False
    if isinstance(sw, dict):
        depths = sw.get("depths") or []
        embeds = sw.get("embed_dims") or []
        ok = (
            isinstance(depths, list)
            and isinstance(embeds, list)
            and len(depths) == len(embeds)
            and len(depths) > 0
        )
    if not ok:
        warns.append(
            VerificationWarning(
                code="SWIN_DIMS_INCOMPLETE",
                message="family='swin'인데 dims.swin.{depths, embed_dims} 구성이 불완전합니다.(기본값으로 보정될 수 있음)",
                fix_applied=False,
            )
        )


@register_family_validator("performer")
def _validate_performer(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    # Expect decoder_layers presence; attn_impl defaults to 'performer' in codegen wrapper
    dec = (
        (spec.dims or type(spec).model_fields["dims"].default_factory()).decoder_layers
        if spec.dims
        else None
    )
    if not isinstance(dec, dict):
        warns.append(
            VerificationWarning(
                code="PERFORMER_DECODER_MISSING",
                message="family='performer'인데 dims.decoder_layers 구조가 없습니다.",
                fix_applied=False,
            )
        )
        return
    nb = dec.get("attn_nb_features")
    if nb is not None:
        try:
            nb = int(nb)
            if nb <= 0:
                raise ValueError
        except Exception:
            warns.append(
                VerificationWarning(
                    code="PERFORMER_NBFEATURES_INVALID",
                    message="attn_nb_features는 양의 정수여야 합니다.",
                    fix_applied=False,
                )
            )


@register_family_validator("performer_mt")
def _validate_performer_mt(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    # MT: also expect a hint that cross-attention is desired
    dec = (
        (spec.dims or type(spec).model_fields["dims"].default_factory()).decoder_layers
        if spec.dims
        else None
    )
    if not isinstance(dec, dict):
        warns.append(
            VerificationWarning(
                code="PERFORMER_MT_DECODER_MISSING",
                message="family='performer_mt'인데 dims.decoder_layers 구조가 없습니다.",
                fix_applied=False,
            )
        )
        return
    if not bool(dec.get("use_cross_attn", True)):
        warns.append(
            VerificationWarning(
                code="PERFORMER_MT_CROSS_ATTN_OFF",
                message="performer_mt에서는 보통 cross-attn을 사용합니다. dims.decoder_layers.use_cross_attn=True를 권장합니다.",
                fix_applied=False,
            )
        )
    nb = dec.get("attn_nb_features")
    if nb is not None:
        try:
            nb = int(nb)
            if nb <= 0:
                raise ValueError
        except Exception:
            warns.append(
                VerificationWarning(
                    code="PERFORMER_NBFEATURES_INVALID",
                    message="attn_nb_features는 양의 정수여야 합니다.",
                    fix_applied=False,
                )
            )


def verify_and_normalize(spec: ModelSpec) -> VerifiedSpec:
    warnings: List[VerificationWarning] = []
    # 공통
    warnings.extend(_common_checks(spec))
    # family별 세부 검증
    validator = FAMILY_VALIDATORS.get((spec.proposed_model_family or "").lower())
    if validator:
        validator(spec, warnings)
    # 소프트 보정
    _soft_coercions(spec, warnings)
    return VerifiedSpec(spec=spec, warnings=warnings)
