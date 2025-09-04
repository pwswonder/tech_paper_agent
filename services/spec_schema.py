"""
services/spec_schema.py

Phase 1 (reboot): 논문에서 '제안 모델' 정보를 구조화하는 표준 스키마.
- 템플릿(ResNet/VGG/DenseNet/Inception/MobileNet/UNet, LSTM/GRU, Transformer, MLP, AE/VAE/GAN 등)을
  모두 커버할 수 있도록 택소노미/차원 필드를 확장했습니다.
- 이후 단계(템플릿 검색·코드 합성·셀프체크)의 단일 진실 소스가 됩니다.
"""

from __future__ import annotations
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------------------
# 0) 택소노미 정의(가급적 '상위 계열'만 family로, 세부는 subtype으로)
# ------------------------------------------------------------
TaskType = Literal[
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
]

DataModality = Literal[
    "time_series", "text", "image", "audio", "tabular", "graph", "multimodal", "other"
]

ModelFamily = Literal[
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
]

# 세부 subtype은 Literal로 강제하지 않고 문자열로 열어 두는 편이 템플릿 확장에 유리
# (예: "ResNet", "VGG", "DenseNet", "Inception", "MobileNet", "UNet",
#      "Encoder", "Decoder", "EncoderDecoder", "Seq2Seq", ...)

ObjectiveType = Literal[
    "mse", "mae", "cross_entropy", "binary_cross_entropy", "huber", "logcosh", "other"
]

PositionalEncodingType = Literal["absolute", "relative", "none", "other"]


# ------------------------------------------------------------
# 1) 보조 스키마
# ------------------------------------------------------------
class BaselineModel(BaseModel):
    name: str = Field(..., description="비교/베이스라인 모델명")
    family: Optional[ModelFamily] = Field(None, description="모델 계열(추정)")
    notes: Optional[str] = Field(None, description="메모")


class EvidenceSnippet(BaseModel):
    text: str = Field(..., description="근거 인용 (짧게)")
    section: Optional[str] = Field(None, description="절 제목")
    page: Optional[int] = Field(None, description="페이지 번호")


class DimensionConfig(BaseModel):
    """
    다양한 모달리티를 위한 공통 차원/하이퍼파라미터.
    - time_series: seq_len, pred_len, in_dim, out_dim, num_heads, ...
    - image: height, width, in_dim(=channels), num_classes(=out_dim)
    - text: vocab_size, max_len

    [세션 확장] 합성기에서 사용하는 구조화 블록을 스키마에 반영:
      - encoder_layers: Dict (Transformer encoder 합성 파라미터)
      - decoder_layers: Dict (Transformer/MT/Performer 디코더 합성 파라미터; attn_nb_features 등 포함)
      - stages: List[Dict] (ResNet/CNN/VGG/DenseNet 등의 stage 표현)
      - vgg: Dict (VGG 전용 파라미터)
      - densenet: Dict (DenseNet 전용 파라미터)
      - swin: Dict (Swin-like 전용 파라미터: depths, embed_dims, window_size, mlp_ratio, patch_merging)
    """

    # 공통
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None
    seq_len: Optional[int] = None
    pred_len: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    ffn_dim: Optional[int] = None
    dropout: Optional[float] = None

    # image/text convenience
    height: Optional[int] = None
    width: Optional[int] = None
    kernel_size: Optional[int] = None
    dilation: Optional[int] = None
    vocab_size: Optional[int] = None
    max_len: Optional[int] = None

    # [NEW] 구조화 합성 블록
    encoder_layers: Optional[Dict] = None
    decoder_layers: Optional[Dict] = None
    stages: Optional[List[Dict]] = None
    vgg: Optional[Dict] = None
    densenet: Optional[Dict] = None
    swin: Optional[Dict] = None


class ModelSpec(BaseModel):
    """
    논문 '제안 모델'의 구조/하이퍼파라미터/근거를 기술하는 메인 스키마.
    """

    # (A) 메타
    title: Optional[str] = None
    task_type: TaskType = "other"
    data_modality: DataModality = "other"

    # (B) 제안 모델 계열/세부유형
    proposed_model_family: ModelFamily = "Other"
    subtype: Optional[str] = Field(
        None, description="세부 유형(예: ResNet, UNet, Encoder, Seq2Seq, ...)"
    )

    # (C) 핵심 블록 시그니처(코드 셀프체크에도 사용)
    #   예: ["Conv2D","Residual","GlobalAveragePooling2D"], ["MultiHeadAttention","LayerNorm"], ["LSTM"], ["GRU"], ...
    key_blocks: List[str] = Field(default_factory=list)

    # (D) 차원/하이퍼파라미터
    dims: DimensionConfig = Field(default_factory=DimensionConfig)

    # (E) 목적함수/포지셔널 인코딩 등
    objective: Optional[ObjectiveType] = None
    positional_encoding: Optional[PositionalEncodingType] = None

    # (F) 비교모델/근거
    baselines: List[BaselineModel] = Field(default_factory=list)
    evidence: List[EvidenceSnippet] = Field(default_factory=list)

    # (G) 신뢰도 & 플래그
    confidence: float = 0.0
    is_proposed_clearly_identified: bool = False

    # Validators
    custom_blocks: Optional[Dict[str, str]] = (
        None  # optional: precomputed/injected CUSTOM_BLOCK sources
    )

    @field_validator("confidence")
    @classmethod
    def _clip_confidence(cls, v: float) -> float:
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    @field_validator("key_blocks")
    @classmethod
    def _normalize_blocks(cls, v: List[str]) -> List[str]:
        out, seen = [], set()
        for s in v:
            s2 = (s or "").strip()
            if s2 and s2.lower() not in seen:
                seen.add(s2.lower())
                out.append(s2)
        return out


class VerificationWarning(BaseModel):
    code: str
    message: str
    fix_applied: bool = False


class VerifiedSpec(BaseModel):
    spec: ModelSpec
    warnings: List[VerificationWarning] = Field(default_factory=list)
